import os
# The real imu_sim2real_plus is one level up from scripts
link_target = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'imu_sim2real_plus'))
link_name = os.path.join(os.path.dirname(__file__), 'imu_sim2real_plus')

if not os.path.exists(link_name):
    os.symlink(link_target, link_name)
    print(f"Created symlink: {link_name} -> {link_target}")
else:
    print("Symlink already exists.")
