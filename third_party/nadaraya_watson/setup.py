# C++ Nadaraya-Watson smoother from CurrOT (https://github.com/psclklnk/currot),
# built with pybind11's setuptools helpers (no CMake required).
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "nadaraya_watson",
        ["interface.cpp"],
        include_dirs=["include"],
        depends=["include/nadaraya_watson.h", "include/nanoflann.hpp"],
        cxx_std=14,
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext}, zip_safe=False)
