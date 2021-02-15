"""
Setup file for progressivis.
"""
import sys
import os
import os.path
import versioneer
from setuptools import setup, Command
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np

CONDA_PREFIX = os.getenv('CONDA_PREFIX')

PACKAGES = ['progressivis',
            'progressivis.utils',
            'progressivis.utils.khash',
            'progressivis.core',
            'progressivis.storage',
            'progressivis.io',
            'progressivis.stats',
            'progressivis.datasets',
            'progressivis.vis',
            'progressivis.cluster',
            'progressivis.metrics',
            'progressivis.server',
            'progressivis.table']

def _pybind11_includes(mode):
    try:
        import pybind11
        return pybind11.get_include(mode)
    except:
        return os.path.join(sys.prefix, 'include')

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
        include_dirs=[np.get_include()],
        extra_compile_args=['-Wfatal-errors'],
    )]

EXT_PYBIND11 = [
    Extension(
        'progressivis.stats.cxx_max',
        ['progressivis/stats/cxx_max.cpp'],
        include_dirs=[
            'include',
            _pybind11_includes(True),
            _pybind11_includes(False),
            np.get_include(),
            os.path.join(sys.prefix, 'include'),
            os.path.join(CONDA_PREFIX, 'include'),
            os.path.join(sys.prefix, 'Library', 'include')
        ],
        #extra_compile_args=['-std=c++17'],
        extra_compile_args=['-std=c++17', '-Wall', '-O0', '-g'],
        extra_link_args=["-lroaring"],
        language='c++'
    ),
] if CONDA_PREFIX else [] # avoids conda dependency ...

def read(fname):
    "Read the content of fname as string"
    with open(os.path.join(os.path.dirname(__file__), fname)) as infile:
        return infile.read()
setup(
    name="progressivis",
    version= '0.0.1', #versioneer.get_version(),
    author="Jean-Daniel Fekete",
    author_email="Jean-Daniel.Fekete@inria.fr",
    url="https://github.com/jdfekete/progressivis",
    description="A Progressive Steerable Analytics Toolkit",
    license="BSD",
    keywords="Progressive analytics visualization",
    packages=PACKAGES,
    long_description=read('README.md'),
    classifiers=["Development Status :: 2 - PRe-Alpha",
                 "Topic :: Scientific/Engineering :: Visualization",
                 "Topic :: Scientific/Engineering :: Information Analysis",
                 "License :: OSI Approved :: BSD License"],
    platforms='any',
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    # install_requires=required,
    install_requires=["Pillow>=4.2.0",
                      "cython",
                      'pybind11>=2.0.1',
                      "numpy>=1.16.5",
                      "scipy>=0.18.1",
                      "numexpr>=2.6.1",
                      "tables>=3.3.0",
                      "pandas>=1.0.0",
                      "scikit-learn>=0.18.1",
                      "tdigest>=0.4.1.0",
                      "numcodecs>=0.5.5",
                      "datashape>=0.5.2",
                      "pyroaring==0.2.9",
                      "msgpack-python>=0.4.8",
                      "python-dateutil>=2.6.1",  # botocore wants < 2.7.0,>=2.1
                      "boto",
                      "s3fs",
                      "sqlalchemy",
                      "memory_profiler",
                      "tabulate",
                      "requests",
                      "fast-histogram",
                      "rangehttpserver",
                      "aiohttp",
                      "aiohttp_jinja2",
                      "python_socketio", "click"],
    # "pptable",
    setup_requires=['cython', 'numpy', 'pybind11', 'nose>=1.3.7', 'coverage'],
    # test_suite='tests',
    test_suite='nose.collector',
    cmdclass=versioneer.get_cmdclass({'bench': RunBench}),
    ext_modules=cythonize(EXTENSIONS) + EXT_PYBIND11,
    package_data={
        # If any package contains *.md, *.txt or *.rst files, include them:
        'doc': ['*.md', '*.rst'],
        }
    )
