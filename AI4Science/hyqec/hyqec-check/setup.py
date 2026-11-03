"""Build standalone C++ executables into a platform-specific Python wheel."""

import os
from pathlib import Path
import shlex
import subprocess

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel


class NativeBuild(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "hyqec_check" / "_native"
        destination.mkdir(parents=True, exist_ok=True)
        compiler = shlex.split(os.environ.get("CXX", "g++"))
        flags = shlex.split(os.environ.get("CXXFLAGS", "-O3 -std=c++20 -Wall -Wextra -Wpedantic"))
        for source, name in (
            ("checker.cpp", "hyqec-check-native-v1.3.0"),
            ("checker_wide.cpp", "hyqec-check-wide-native-v1.3.0"),
            ("reference.cpp", "hyqec-check-reference"),
            ("checker_v2.cpp", "hyqec-check-v2-native-v2.0.0"),
        ):
            subprocess.run(compiler + flags + ["-fopenmp", f"native/{source}", "-o", str(destination / name)],
                           check=True)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        # No Python ABI is used by these standalone executables.
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(distclass=BinaryDistribution,
      cmdclass={"build_py": NativeBuild, "bdist_wheel": PlatformWheel})
