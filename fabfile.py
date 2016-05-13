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
env.project_name = 'progressivis'
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
    print ('\nDevelopment environment successfully created.')

def make_virtual_env():
    "Make a virtual environment for local dev use"
    with lcd(PROJ_ROOT):
        local('virtualenv .')
        local('./bin/pip install -r requirements.txt')

