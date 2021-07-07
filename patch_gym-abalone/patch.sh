pip uninstall gym-abalone
cp -vf patch_gym-abalone/abalone_env.py gym-abalone/gym_abalone/envs/abalone_env.py
cd gym-abalone
pip install -e .