from setuptools import setup, find_packages

setup(
    name="causal_gym",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "networkx",
    ],
)