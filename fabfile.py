from fabric.api import env, local, lcd
from fabric.colors import red, green
from fabric.decorators import task, runs_once
from fabric.operations import prompt
from fabric.utils import abort
from zipfile import ZipFile

import datetime
import fileinput
import importlib
import os
import random
import re
import subprocess
import sys
import time

PROJ_ROOT = os.path.dirname(env.real_fabfile)
env.project_name = 'progressive-mds'
env.python = 'python' if 'VIRTUAL_ENV' in os.environ else './bin/python'

@task
def setup():
    """
    Set up a local development environment

    This command must be run with Fabric installed globally (not inside a
    virtual environment)
    """
    if os.getenv('VIRTUAL_ENV') or hasattr(sys, 'real_prefix'):
        abort(red('Deactivate any virtual environments before continuing.'))
    make_virtual_env()
    #install_node_packages()
    symlink_packages()
    print ('\nDevelopment environment successfully created.')

@task
def runserver():
    "Run the development server"
    with lcd(PROJ_ROOT):
        local('{python} runserver.py'.format(**env))

def make_virtual_env():
    "Make a virtual environment for local dev use"
    with lcd(PROJ_ROOT):
        local('virtualenv .')
        local('./bin/pip install -r requirements.txt')

def install_node_packages():
    "Install requirements from NPM."
    with lcd(PROJ_ROOT):
        local('npm install')


def symlink_packages():
    "Symlink python packages not installed with pip"
    missing = []
    requirements = (req.rstrip().replace('# symlink: ', '')
                    for req in open('requirements.txt', 'r')
                    if req.startswith('# symlink: '))
    for req in requirements:
        try:
            module = importlib.import_module(req)
        except ImportError:
            missing.append(req)
            continue
        with lcd(os.path.join(PROJ_ROOT, 'lib', 'python2.7', 'site-packages')):
            local('ln -f -s {}'.format(os.path.dirname(module.__file__)))
    if missing:
        abort('Missing python packages: {}'.format(', '.join(missing)))

@task
def runserver():
    "Run the development server"
    with lcd(PROJ_ROOT):
        local('{python} app.py'.format(**env))
