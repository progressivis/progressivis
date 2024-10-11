# type: ignore
"""
Setup file for progressivis.
"""
import sys
import os
import os.path

# import versioneer
from setuptools import setup, Command
from setuptools.extension import Extension

CONDA_PREFIX = os.getenv("CONDA_PREFIX", "")
MYBINDER = os.getenv("USER") == "jovyan"
WITH_CXX = (not MYBINDER) and sys.platform == "linux"


PACKAGES = [
    "progressivis",
    "progressivis.utils",
    "progressivis.core",
    "progressivis.storage",
    "progressivis.io",
    "progressivis.stats",
    "progressivis.datasets",
    "progressivis.datashape",
    "progressivis.datashape.util",
    "progressivis.vis",
    "progressivis.cluster",
    "progressivis.server",
    "progressivis.table",
]


def _cythonize(exts):
    from Cython.Build import cythonize

    return cythonize(exts)


def _np_get_include():
    import numpy as np

    return np.get_include()


class RunBench(Command):
    """Runs all ProgressiVis benchmarks"""

    description = "run all benchmarks"
    user_options = []  # distutils complains if this is not here.

    def __init__(self, *args):
        self.args = args[0]  # so we can pass it to other classes
        Command.__init__(self, *args)

    def initialize_options(self):
        "distutils wants this"
        pass

    def finalize_options(self):
        "distutils wants this"
        pass

    def run(self):
        "Run the benchark"
        for root, _, files in os.walk("benchmarks"):
            for fname in files:
                if fname.startswith("bench_") and fname.endswith(".py"):
                    pathname = os.path.join(root, fname)
                    self._run_it(pathname)

    def _run_it(self, pathname):
        if self.verbose:  # verbose is provided "automagically"
            print('Should be running bench "{0}"'.format(pathname))
        # TODO run the command with the right arguments


EXTENSIONS = [
    Extension(
        "progressivis.utils.fast",
        ["progressivis/utils/fast.pyx"],
        include_dirs=[_np_get_include()],
        extra_compile_args=["-Wfatal-errors"],
    )
]

EXT_PYBIND11 = [
    Extension(
        "progressivis.stats.cxx_max",
        ["progressivis/stats/cxx_max.cpp"],
        include_dirs=[
            "include",
            _np_get_include(),
            "pybind11/include",
            "xtensor/include",
            "xtensor-python/include",
            "xtl/include",
            "CRoaringUnityBuild",
            os.path.join(sys.prefix, "include"),
            os.path.join(CONDA_PREFIX, "include"),
            os.path.join(sys.prefix, "Library", "include"),
        ],
        extra_compile_args=["-std=c++17", "-Wall", "-O0", "-g"],
        language="c++",
    ),
]


def read(fname):
    "Read the content of fname as string"
    with open(os.path.join(os.path.dirname(__file__), fname)) as infile:
        return infile.read()


setup(
    url="https://github.com/progressivis/progressivis",
    packages=PACKAGES,
    platforms="any",
    ext_modules=_cythonize(EXTENSIONS) + EXT_PYBIND11 if WITH_CXX else [],
    package_data={
        # If any package contains *.md, *.txt or *.rst files, include them:
        "doc": ["*.md", "*.rst"],
        "progressivis": ["py.typed"],
    },
)
