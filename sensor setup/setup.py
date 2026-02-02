from setuptools import setup, Extension
import pybind11
import os

# Compiler arguments for optimization and C++ standard
cpp_args = ['-std=c++14', '-O3']

ext_modules = [
    Extension(
        'sim2real_native',              # The name of the module to import in Python
        ['sensor_noise.cpp'],           # The source C++ file
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='sim2real_native',
    version='0.1',
    description='C++ Accelerated Noise Injection for Isaac Sim',
    ext_modules=ext_modules,
)