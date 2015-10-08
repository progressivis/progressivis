import logging
log = logging.getLogger(__name__)

from flask import (
    Flask, Blueprint,
    render_template, request, send_from_directory,
    abort, jsonify, Response, redirect, url_for
)

import sys
from os.path import join, dirname, abspath, normpath, realpath, isdir

from progressivis.core.scheduler import Scheduler

class ProgressivisBlueprint(Blueprint):
    def __init__(self, *args, **kwargs):
        super(ProgressivisBlueprint, self).__init__(*args, **kwargs)

    def setup(self, scheduler):
        self.scheduler = scheduler

#    
progressivis_bp = ProgressivisBlueprint('progressivis.server',
                                        'progressivis.server',
                                        static_folder='static',
                                        static_url_path='/progressivis/static',
                                        template_folder='templates')

import views


def create_app(config="settings.py", scheduler=None):
    if scheduler is None:
        scheduler = Scheduler.default
    app = Flask('progressivis.server')
    if isinstance(config, str):
        app.config.from_pyfile(config)
    elif isinstance(config, dict):
        app.config.update(config)

    app.register_blueprint(progressivis_bp)
    progressivis_bp.setup(scheduler)

    return app

