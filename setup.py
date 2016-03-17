import os
import pkg_resources
from setuptools import setup, find_packages

execfile('progressivis/core/version.py')

#with open('requirements.txt') as f:
#    required = f.read().splitlines()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "progressivis",
    version = __version__,
    author = "Jean-Daniel Fekete",
    author_email = "Jean-Daniel.Fekete@inria.fr",
    url="http://progressive.gforge.inria.fr/",
    description = "A Progressive Steerable Analytics Toolkit",
    license = "BSD",
    keywords = "Progressive analytics visualization",
    packages = ['progressivis',
                'progressivis.core',
                'progressivis.io',
                'progressivis.stats',
                'progressivis.datasets',
                'progressivis.vis',
                'progressivis.cluster',
                'progressivis.manifold',
                'progressivis.metrics',
                ],
    long_description = read('README.md'),
    classifiers=[
        "Development Status :: 2 - PRe-Alpha",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: BSD License",
    ],
    platforms='any',
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    #install_requires = required,
    install_requires = [
        "Pillow==3.1.1",
        "numpy==1.10.4",
        "scipy==0.17.0",
        "numexpr==2.4.6",
        "cython==0.23.4",
        "tables==3.2.2",
        "pandas==0.18.0",
        "scikit-learn==0.17.1",
        "toposort==1.4",
        "tdigest==0.4.0.1",
        "flask==0.10.1",
        "tornado==4.3",
    ],
    test_suite='tests',

    package_data = {
        # If any package contains *.md, *.txt or *.rst files, include them:
        'doc': ['*.md', '*.rst'],
    },

)
