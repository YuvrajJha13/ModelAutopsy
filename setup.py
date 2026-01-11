from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import sys

# Compiler Flags
if sys.platform == "win32":
    copt, lopt = ['/O2', '/std:c++17', '/openmp'], []
else:
    copt, lopt = ['-O3', '-march=native', '-std=c++17', '-fopenmp'], ['-fopenmp']

ext_modules = [
    Pybind11Extension(
        "_core_cpp",
        sources=["src/module.cpp", "src/stats.cpp"],
        include_dirs=["include"],
        extra_compile_args=copt,
        extra_link_args=lopt,
    ),
]

setup(
    name="modelautopsy",
    version="1.0.0",
    author="Yuvraj Jha",
    description="ModelAutopsy: High-Performance ML Failure Detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YuvrajJha13/MLGuardian",
    packages=find_packages(exclude=("tests", "examples")),
    ext_modules=ext_modules,
    install_requires=["numpy", "rich"],
    setup_requires=["pybind11"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
