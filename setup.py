import os
import pkg_resources
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "progressivis",
    version = pkg_resources.require("progressivis")[0].version, # '0.0.2.dev0'
    author = "Jean-Daniel Fekete",
    author_email = "Jean-Daniel.Fekete@inria.fr",
    url="http://progressive.gforge.inria.fr/",
    description = "A Progressive Steerable Analytics Toolkit",
    license = "BSD",
    keywords = "Progressive analytics visualization",
    packages = ['progressivis'],
    long_description = read('README.md'),
    classifiers=[
        "Development Status :: 2 - PRe-Alpha",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: BSD License",
    ],
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = required,
    test_suite='tests',

    package_data = {
        # If any package contains *.md, *.txt or *.rst files, include them:
        '': ['*.md', '*.rst'],
    },

)
