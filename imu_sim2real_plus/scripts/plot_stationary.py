import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='runs/stationary_vibration', help='Directory containing the sequence data')
    args = parser.parse_args()

    f_meas_path = os.path.join(args.input_dir, 'seq_00000_f_meas.npy')
    w_meas_path = os.path.join(args.input_dir, 'seq_00000_w_meas.npy')

    if not os.path.exists(f_meas_path) or not os.path.exists(w_meas_path):
        print(f"Could not find sequence data in {args.input_dir}")
        # Try to generate it
        print("Generating data...")
        try:
            from scripts.generate_stationary_vibration import main as generate_main
            # mock args
            class MockArgs:
                def __init__(self):
                    self.config = 'imu_sim2real_plus/config/example_config.yaml'
                    self.out = args.input_dir
                    self.seconds = 10.0
                    self.odr = 400
                    self.rpm = 1000.0
            generate_main(MockArgs())
        except Exception as e:
            print(f"Failed to generate data: {e}")
            return

    f_meas = np.load(f_meas_path)
    w_meas = np.load(w_meas_path)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(f_meas)
    axs[0].set_title('Accelerometer Measurements')
    axs[0].set_ylabel('m/s^2')
    axs[0].legend(['x', 'y', 'z'])
    axs[0].grid(True)

    axs[1].plot(w_meas)
    axs[1].set_title('Gyroscope Measurements')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('rad/s')
    axs[1].legend(['x', 'y', 'z'])
    axs[1].grid(True)

    plt.tight_layout()
    
    plot_path = os.path.join(args.input_dir, 'stationary_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    main()
