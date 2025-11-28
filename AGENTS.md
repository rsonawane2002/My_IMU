# Repository Guidelines

## Project Structure & Module Organization
- `imu_sim2real_plus/`: main package.
  - `scripts/`: CLI entry points (e.g., `generate_dataset.py`, `run_allan.py`).
  - `sensors/`, `metrics/`, `models/`, `dataio/`: core logic by domain.
  - `config/`: YAML configs and `schema.py` (e.g., `example_config.yaml`).
  - `sim/`, `ros2/`: integration shims for simulators/ROS 2.
  - `tests/`: Pytest tests (e.g., `test_allan.py`, `test_synth.py`).
- Root utilities: plotting/analysis scripts (e.g., `plot_imu_data.py`, `inject_noise_from_log.py`).
- Artifacts/output: `runs/` (not for version control).

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`.
- Install deps: `pip install -r requirements.txt`.
- Generate data: `python -m imu_sim2real_plus.scripts.generate_dataset --config imu_sim2real_plus/config/example_config.yaml --out runs/sim --sequences 3 --seconds 10 --odr 400`.
- Allan analysis: `python -m imu_sim2real_plus.scripts.run_allan --series runs/sim/seq_00000_f_meas.npy --fs 400`.
- Run tests (pytest): `python -m pytest -q` (install with `pip install pytest` if needed).
 - Inject noise from CSV log: `python inject_noise_from_log.py --log_file trajectory.csv --config imu_sim2real_plus/config/example_config.yaml --out runs/log_noise`.

## Log Noise Injection
- Input CSV expects time as first column and headers: `acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z, quat_w, quat_x, quat_y, quat_z`.
- The script converts quaternions to `R_WB`, computes `a_W`, synthesizes IMU via `synth_measurements`, and writes a sequence to `--out` using `save_sequence`.
- Keep outputs in `runs/` and avoid committing large artifacts.

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indent; line length ≈ 88.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Type hints welcomed. Keep modules focused and colocate helpers near domain folders (`sensors/`, `metrics/`, etc.).
- Config in YAML; validate with `config/schema.py` if you add keys.

## Testing Guidelines
- Framework: Pytest; put tests in `imu_sim2real_plus/tests/` as `test_*.py`.
- Cover new logic with unit tests close to the domain (e.g., sensor models in `tests` touching `sensors/`).
- Keep tests deterministic (seed RNGs) and fast; avoid writing outside `runs/`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject; include context in the body and reference issues (`Fixes #123`).
- Scope small and atomic; separate refactors from behavior changes.
- PRs: clear description, motivation, and screenshots/plots for analysis changes; list commands used to reproduce.
- Add/update tests and configs; note any data or schema changes.

## Security & Configuration Tips
- Do not commit large artifacts or proprietary logs; keep outputs in `runs/`.
- Parameterize paths via YAML; avoid hard‑coded local paths.
- When integrating simulators/ROS, mock interfaces in tests.
