#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <iostream>
#include <algorithm> 

namespace py = pybind11;
using namespace pybind11::literals;

// --- Helper Structures ---
struct ResonantMode{
    float f0;
    float zeta;
    float gain;
    std::vector<int> axes;
};

// --- Filter Class ---
class StatefulFilter{
public:
    std::vector<double> b;
    std::vector<double> a;
    std::vector<double> zi;

    StatefulFilter() {} 

    StatefulFilter(std::vector<double> b_coeffs, std::vector<double> a_coeffs): b(b_coeffs), a(a_coeffs)
    {
        zi.assign(std::max(b.size(), a.size()) -1, 0.0);
    }

    double step(double x) {
        if (a.empty()) return x; 
        double y = b[0] * x + zi[0];
        for (size_t i = 0; i < zi.size() - 1; ++i) {
            zi[i] = b[i+1] * x - a[i+1] * y + zi[i+1];
        }
        zi.back() = b.back() * x - a.back() * y;
        return y;
    }
};

std::pair<std::vector<double>, std::vector<double>> bilinear_resonator(double fs, const ResonantMode& mode) {
    double wn = 2 * M_PI * mode.f0;
    double zeta = mode.zeta;
    double A = 2.0 * zeta * wn * mode.gain; 
    double C = 2.0 * zeta * wn;             
    double D = wn * wn;                     
    double K = 2.0 * fs;
    double K2 = K * K;
    double norm = K2 + C * K + D;
    double b0 = (A * K) / norm;
    double b1 = 0.0;           
    double b2 = -(A * K) / norm;
    double a0 = 1.0; 
    double a1 = (2.0 * D - 2.0 * K2) / norm;
    double a2 = (K2 - C * K + D) / norm;
    return {{b0, b1, b2}, {a0, a1, a2}};
}

// --- Main Core Class ---
class Sim2RealCore{
private:
    std::mt19937 rng;
    bool filters_initialized;
    double last_time;

    // Internal State
    double motor_phase;
    double arma_prev_y;
    double arma_prev_e;
    std::vector<double> gyro_bias;

    // Filters
    std::map<int, StatefulFilter> filters_acc;
    std::map<int, StatefulFilter> filters_gyr;

    // --- CONFIGURATION UPDATES ---
    
    // 1. High-Level Knobs (UPDATED)
    double accel_fs_g = 2.0;       // [UPDATED] Was 8.0, now 2.0 per YAML
    double gyro_fs_dps = 250.0;    // [UPDATED] Was 2000.0, now 250.0 per YAML
    double odr_hz = 104.0;         // Confirmed 104Hz
    bool use_vibration = false;    

    // 2. Vibration Model Internals
    std::vector<ResonantMode> modes_acc;
    std::vector<ResonantMode> modes_gyr;
    std::map<int, double> motor_harmonics;
    double floor_noise_sigma = 0.4;
    double floor_noise_ar = 0.96;
    double floor_noise_ma = 0.2;
    double g_sensitivity = 0.002;
    
    // 3. Noise & Scaling Defaults (UPDATED)
    // Calculated as: Density * sqrt(ODR)
    double accel_white_std = 0.0060;   // [UPDATED] Was 0.2
    double gyro_white_std = 0.00089;   // [UPDATED] Was 0.0025
    double gyro_bias_rw_std = 2e-5;    // Kept low for stability
    
    // [UPDATED] Was {1.0, 0.6, 0.9}. Reset to 1.0 because 'scale_ppm' is tight (~0.3%)
    std::vector<double> axis_scale = {1.0, 1.0, 1.0}; 
    
    int quantization_bits = 16; 

    // Helper: Quantize value
    double quantize(double val, double q) {
        if (q <= 1e-9) return val; 
        return std::round(val / q) * q;
    }

public:
    Sim2RealCore(int seed) : filters_initialized(false), last_time(-1.0), motor_phase(0.0), arma_prev_y(0.0), arma_prev_e(0.0) {
        rng.seed(seed);
        gyro_bias = {0.0, 0.0, 0.0};

        // --- PRESERVED "QUIET" VIBRATION DEFAULTS ---
        // These match the working "Low Noise" setup we verified earlier
        modes_acc = {
            {75.0, 0.02, 0.05, {0,1,2}}, 
            {95.0, 0.03, 0.02, {0,1,2}},
            {120.0, 0.03, 0.04, {2}},    
            {130.0, 0.03, 0.03, {2}}
        };
        modes_gyr = {
            {6.5, 0.07, 0.02, {0,1,2}},  
            {8.0, 0.08, 0.015, {0,1,2}},
            {75.0, 0.03, 0.004, {2}},    
            {120.0, 0.03, 0.004, {2}}
        };
        motor_harmonics = {{1, 1.0}, {2, 0.35}, {3, 0.2}};
    }

    // --- ROBUST SETTER ---
    void update_configuration(py::dict config) {
        
        // 1. High Level Knobs
        if (config.contains("accel_fs_g"))  accel_fs_g  = config["accel_fs_g"].cast<double>();
        if (config.contains("gyro_fs_dps")) gyro_fs_dps = config["gyro_fs_dps"].cast<double>();
        if (config.contains("odr_hz"))      odr_hz      = config["odr_hz"].cast<double>();
        if (config.contains("vibration"))   use_vibration = config["vibration"].cast<bool>();

        // 2. Vibration Internals
        if (config.contains("g_sensitivity")) g_sensitivity = config["g_sensitivity"].cast<double>();
        if (config.contains("floor_noise_sigma")) floor_noise_sigma = config["floor_noise_sigma"].cast<double>();
        
        filters_initialized = false; 
    }

    void init_filters(double dt) {
        if (dt <= 0) return;
        double fs = 1.0 / dt;
        filters_acc.clear();
        filters_gyr.clear();
        
        for(size_t i=0; i<modes_acc.size(); ++i) {
            auto coeffs = bilinear_resonator(fs, modes_acc[i]);
            filters_acc[i] = StatefulFilter(coeffs.first, coeffs.second);
        }
        for(size_t i=0; i<modes_gyr.size(); ++i) {
            auto coeffs = bilinear_resonator(fs, modes_gyr[i]);
            filters_gyr[i] = StatefulFilter(coeffs.first, coeffs.second);
        }
        filters_initialized = true;
    }

    py::dict process(py::array_t<double> true_acc_np, py::array_t<double> true_ang_np, double current_time) 
    {
        auto true_acc = true_acc_np.unchecked<1>();
        auto true_ang = true_ang_np.unchecked<1>();

        double dt_sim = 0.0;
        if (last_time >= 0) dt_sim = current_time - last_time;
        last_time = current_time;

        double dt_calc = (odr_hz > 0.0) ? (1.0 / odr_hz) : dt_sim;

        if (dt_calc <= 1e-6) {
            py::dict res; res["lin_acc"] = true_acc_np; res["ang_vel"] = true_ang_np; return res;
        }

        if (!filters_initialized) init_filters(dt_calc);

        std::vector<double> vib_accel = {0, 0, 0};
        std::vector<double> vib_gyro = {0, 0, 0};
        
        if (use_vibration) {
            double rpm = 4500.0;
            motor_phase += 2 * M_PI * (rpm / 60.0) * dt_calc;
            
            double exc = 0.0;
            for (auto const& [harmonic, amp] : motor_harmonics) {
                exc += amp * std::sin(harmonic * motor_phase);
            }

            std::normal_distribution<double> dist_norm(0.0, 1.0);
            double current_e = dist_norm(rng) * floor_noise_sigma;
            double floor = floor_noise_ar * arma_prev_y + current_e + floor_noise_ma * arma_prev_e;
            arma_prev_y = floor;
            arma_prev_e = current_e;

            double base_exc = exc + 0.25 * floor;

            for(size_t i=0; i<modes_acc.size(); ++i) {
                double val = filters_acc[i].step(base_exc);
                for(int ax : modes_acc[i].axes) vib_accel[ax] += axis_scale[ax] * val;
            }
            for(size_t i=0; i<modes_gyr.size(); ++i) {
                double val = filters_gyr[i].step(base_exc);
                for(int ax : modes_gyr[i].axes) vib_gyro[ax] += axis_scale[ax] * val;
            }
        }

        std::normal_distribution<double> dist_norm(0.0, 1.0);
        std::vector<double> n_acc(3), n_gyr(3);
        
        for(int i=0; i<3; ++i) {
            // Apply Calculated White Noise
            n_acc[i] = dist_norm(rng) * accel_white_std;
            n_gyr[i] = dist_norm(rng) * gyro_white_std;
            gyro_bias[i] += dist_norm(rng) * (gyro_bias_rw_std * std::sqrt(dt_calc));
        }

        // Summation, Clipping, Quantization
        double qa = (2.0 * accel_fs_g * 9.80665) / std::pow(2, quantization_bits);
        double gyro_fs_rad = gyro_fs_dps * (M_PI / 180.0);
        double qg = (2.0 * gyro_fs_rad) / std::pow(2, quantization_bits);
        double acc_lim = accel_fs_g * 9.80665;
        double gyr_lim = gyro_fs_rad;

        std::vector<double> final_acc(3), final_ang(3);
        for(int i=0; i<3; ++i) {
            double raw_a = true_acc(i) + vib_accel[i] + n_acc[i];
            double coupling = g_sensitivity * vib_accel[i];
            double raw_g = true_ang(i) + vib_gyro[i] + coupling + n_gyr[i] + gyro_bias[i];

            double clipped_a = std::max(-acc_lim, std::min(raw_a, acc_lim));
            double clipped_g = std::max(-gyr_lim, std::min(raw_g, gyr_lim));

            if (quantization_bits < 32) {
                final_acc[i] = quantize(clipped_a, qa);
                final_ang[i] = quantize(clipped_g, qg);
            } else {
                final_acc[i] = clipped_a;
                final_ang[i] = clipped_g;
            }
        }

        return py::dict("lin_acc"_a = py::array(3, final_acc.data()), 
                        "ang_vel"_a = py::array(3, final_ang.data()));
    }
};

PYBIND11_MODULE(sim2real_native, m) {
    py::class_<Sim2RealCore>(m, "Sim2RealCore")
        .def(py::init<int>())
        .def("process", &Sim2RealCore::process)
        .def("update_configuration", &Sim2RealCore::update_configuration); 
}