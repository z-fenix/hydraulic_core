#!/usr/bin/env python3
"""
hydro_core — Python package installer

Usage:
    pip install -e .                    # editable install (dev)
    pip install .                       # regular install
    python setup.py build_ext --inplace # build extension only
    python setup.py install             # install

Requirements:
    pybind11, numpy, netcdf4 (libnetcdf-dev)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

from setuptools import setup, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()
HYDRO_PKG = HERE / "hydro"

# ---------------------------------------------------------------------------
# Custom build command for pybind11 extension
# ---------------------------------------------------------------------------

class BuildPybind11(Command):
    """Build the hydro._core pybind11 extension using the standalone script."""

    description = "build pybind11 _core extension"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        script = HERE / "scripts" / "build_pybind11.sh"
        if script.exists():
            print(f"[hydro] Running {script} install ...")
            subprocess.check_call([str(script), "install"])
        else:
            # Fallback: try cmake
            print("[hydro] build script not found, trying cmake ...")
            build_dir = HERE / "build_cmake"
            build_dir.mkdir(exist_ok=True)
            subprocess.check_call(
                ["cmake", str(HERE),
                 "-DHYDRO_BUILD_PYBIND11=ON",
                 "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"],
                cwd=str(build_dir))
            subprocess.check_call(
                ["make", "-j", str(os.cpu_count() or 4), "_core"],
                cwd=str(build_dir))
            # Copy .so to package dir
            for so in build_dir.glob("_core*.so"):
                shutil.copy2(str(so), str(HYDRO_PKG / so.name))
                print(f"[hydro] Copied {so.name} → {HYDRO_PKG}/")


class BuildExt(build_ext):
    """Hook the pybind11 build into setuptools build_ext."""

    def run(self):
        self.run_command("build_pybind11")
        # Skip the default C extension build (pybind11 handles it)
        build_ext.run(self)


class Install(install):
    """Hook pybind11 build into install."""

    def run(self):
        self.run_command("build_pybind11")
        install.run(self)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup(
    name="hydro_core",
    version="0.1.0",
    description="Shallow water equation solver — standalone C library with Python bindings",
    author="ANUGA Community",
    url="https://github.com/anuga-community/hydro_core",
    license="BSD",

    packages=["hydro"],
    package_dir={"hydro": "hydro"},
    package_data={"hydro": ["_core*.so"]},

    python_requires=">=3.10",
    install_requires=["numpy>=2.0", "pybind11>=3.0"],
    extras_require={"netcdf": ["netCDF4"]},

    cmdclass={
        "build_pybind11": BuildPybind11,
        "build_ext": BuildExt,
        "install": Install,
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Programming Language :: C++",
    ],
)
