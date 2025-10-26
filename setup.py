from setuptools import setup

setup(
    name="causal_rl",
    version="1.0.0",
    install_requires=[
        "gymnasium>=0.29.1", 
        "pygame>=2.5.2", 
        "multiprocess>=0.70.16",
        "highway-env>=1.4.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5,<3.10",
        "minigrid",
        "gymnasium-robotics>=1.2",
        "pybullet",
        "mujoco_py"
        # "causal_gym>=1.0.0"
    ],
)
