"""
Setup file for progressivis.
"""
import os
import os.path
import versioneer
from setuptools import setup, Command
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

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
# 'stool'


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
    ),
    Extension("progressivis.utils.khash.hashtable",
              ["progressivis/utils/khash/hashtable.pyx"],
              include_dirs=['progressivis/utils/khash/klib',
                            'progressivis/utils/khash',
                            np.get_include()],
              extra_compile_args=['-Wfatal-errors'])]


def read(fname):
    "Read the content of fname as string"
    with open(os.path.join(os.path.dirname(__file__), fname)) as infile:
        return infile.read()


setup(
    name="progressivis",
    version=versioneer.get_version(),
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
                      "numpy>=1.11.3",
                      "scipy>=0.18.1",
                      "numexpr>=2.6.1",
                      "tables>=3.3.0",
                      "pandas>=0.19.1",
                      "scikit-learn>=0.18.1",
                      "tdigest>=0.4.1.0",
                      "flask>=0.12.1",
                      "eventlet>=0.22.0",
                      "flask-socketio>=2.9.0",
                      "numcodecs>=0.5.5",
                      "datashape>=0.5.2",
                      "pyroaring>=0.2.3",
                      "msgpack-python>=0.4.8",
                      "python-dateutil==2.6.1",  # botocore wants < 2.7.0,>=2.1
                      "boto",
                      "s3fs",
                      "sqlalchemy",
                      "memory_profiler",
                      "tabulate",
                      "requests",
                      "fast-histogram",
                      "rangehttpserver"],
    # "pptable",
    setup_requires=['cython', 'numpy', 'nose>=1.3.7', 'coverage'],
    # test_suite='tests',
    test_suite='nose.collector',
    cmdclass=versioneer.get_cmdclass({'bench': RunBench}),
    ext_modules=cythonize(EXTENSIONS),
    package_data={
        # If any package contains *.md, *.txt or *.rst files, include them:
        'doc': ['*.md', '*.rst'],
        }
    )
