#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <iostream>

namespace py = pybind11;

//Data structures
struct ResonantMode{
    float f0;
    float zeta;
    float gain;
    std::vector<int> axes;
};


//Define our StatefulFilter class (adapted from inject_vibration and imu_sensor_vibration)
class StatefulFilter{
public:
    std::vector<double> b;
    std::vector<double> a;
    //state buffer, to calculate states between frames for scipy signal
    std::vector<double> zi;

    StatefulFilter() {} // default constructor

    StatefulFilter(std::vector<double> b_coeffs, std::vector<double> a_coeffs): b(b_coeffs), a(a_coeffs)
    {

        //we only need to remember n-1 states. So based on max a/b size we store
        //only n-1 states
        zi.assign(std::max(b.size(), a.size()) -1, 0.0);
    }

    //scipy.signal interpration in C++. prevents context switching + overhead
    double step(double x) {
        if (a.empty()) return x; 

        // y[n] = b[0]*x[n] + z[0]
        double y = b[0] * x + zi[0];

        // Update state
        // zi[0] = b[1]*x[n] - a[1]*y[n] + zi[1]
        // zi[1] = b[2]*x[n] - a[2]*y[n]
        for (size_t i = 0; i < zi.size() - 1; ++i) {
            zi[i] = b[i+1] * x - a[i+1] * y + zi[i+1];
        }
        zi.back() = b.back() * x - a.back() * y;

        return y;
    }
};


//bilinear_resonator: adapted from original combined script. Not using scipy here as it is overkill
// and we only need 2nd order polynomial.
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

class Sim2RealCore{
private:
    std::mt19937 rng;
    bool filters_initialized;
    double last_time;

    //state
    double motor_phase;
    double arma_prev_y;
    double arma_prev_e;
    std::vector<double> gyro_bias;


    //Filters
    std::map<int, StatefulFilter> filters_acc;
    std::map<int, StatefulFilter> filters_gyr;


    //config
    std::vector<ResonantMode> modes_acc;
    std::vector<ResonantMode> modes_gyr;
    double accel_white_std = 0.06;
    double gyro_white_std = 0.0025;
    double gyro_bias_rw_std = 2e-5;
    double g_sensitivity = 0.002;
    std::vector<double> axis_scale = {1.0, 0.6, 0.9};

public:


    //construct class with seed
    Sim2RealCore(int seed) : filters_initialized(false), last_time(-1.0), motor_phase(0.0), arma_prev_y(0.0), arma_prev_e(0.0) {
        rng.seed(seed);
        gyro_bias = {0.0, 0.0, 0.0};

        // Define Modes (Ported from Python)
        modes_acc = {
            {75.0, 0.02, 0.05, {0,1,2}}, {95.0, 0.03, 0.02, {0,1,2}},
            {120.0, 0.03, 0.04, {2}},    {130.0, 0.03, 0.03, {2}}
        };
        modes_gyr = {
            {6.5, 0.07, 0.02, {0,1,2}},  {8.0, 0.08, 0.015, {0,1,2}},
            {75.0, 0.03, 0.004, {2}},    {120.0, 0.03, 0.004, {2}}
        };
    }


    //initialize the filters for adding noise
    void init_filters(double dt) {
        if (dt <= 0) return;
        double fs = 1.0 / dt;

        // Clean usage of the helper function!
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

    // Main processing function: This will be invoked from sim to convert raw data into noisy data
    py::dict process(py::array_t<double> true_acc_np, py::array_t<double> true_ang_np, double current_time) 
    {
        auto true_acc = true_acc_np.unchecked<1>();
        auto true_ang = true_ang_np.unchecked<1>();

        double dt = 0.0;
        if (last_time >= 0) 
        {
            dt = current_time - last_time;
        }

        last_time = current_time;

        if (dt <= 1e-6 || !filters_initialized) 
        {
            if (dt > 1e-6 && !filters_initialized) init_filters(dt);
            // Return clean data if not ready
            py::dict res;
            res["lin_acc"] = true_acc_np;
            res["ang_vel"] = true_ang_np;
            return res;
        }

        // --- Noise Logic ---
        
        // A. Motor Excitation
        double rpm = 4500.0;
        motor_phase += 2 * M_PI * (rpm / 60.0) * dt;
        double exc = 1.0 * std::sin(1 * motor_phase) + 0.4 * std::sin(2 * motor_phase) + 0.2 * std::sin(3 * motor_phase);

        // B. ARMA Noise
        std::normal_distribution<double> dist_norm(0.0, 1.0);
        double current_e = dist_norm(rng) * 0.4; // sigma=0.4
        double floor = 0.96 * arma_prev_y + current_e + 0.2 * arma_prev_e;
        arma_prev_y = floor;
        arma_prev_e = current_e;

        // C. Base Excitation
        double base_exc = exc + 0.25 * floor;

        // D. Filter Banks
        std::vector<double> vib_accel = {0, 0, 0};
        std::vector<double> vib_gyro = {0, 0, 0};

        // Accel
        for(size_t i=0; i<modes_acc.size(); ++i) {
            double val = filters_acc[i].step(base_exc);
            for(int ax : modes_acc[i].axes) vib_accel[ax] += axis_scale[ax] * val;
        }
        // Gyro
        for(size_t i=0; i<modes_gyr.size(); ++i) {
            double val = filters_gyr[i].step(base_exc);
            for(int ax : modes_gyr[i].axes) vib_gyro[ax] += axis_scale[ax] * val;
        }

        // E. Noise & Coupling
        std::vector<double> n_acc(3), n_gyr(3);
        for(int i=0; i<3; ++i) {
            n_acc[i] = dist_norm(rng) * accel_white_std;
            n_gyr[i] = dist_norm(rng) * gyro_white_std;
            
            // Random Walk Update
            gyro_bias[i] += dist_norm(rng) * (gyro_bias_rw_std * std::sqrt(dt));
        }

        // F. Final Summation
        std::vector<double> final_acc(3), final_ang(3);
        for(int i=0; i<3; ++i) {
            final_acc[i] = true_acc(i) + vib_accel[i] + n_acc[i];
            
            double coupling = g_sensitivity * vib_accel[i];
            final_ang[i] = true_ang(i) + vib_gyro[i] + coupling + n_gyr[i] + gyro_bias[i];
        }

        py::dict results;
        results["lin_acc"] = py::array(3, final_acc.data());
        results["ang_vel"] = py::array(3, final_ang.data());

        return results;
    }

};

PYBIND11_MODULE(sim2real_native, m) {
    py::class_<Sim2RealCore>(m, "Sim2RealCore")
        .def(py::init<int>())
        .def("process", &Sim2RealCore::process);
}

