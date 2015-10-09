from __future__ import print_function

import logging
log = logging.getLogger(__name__)

from flask import (
    Flask, Blueprint,
    render_template, request, send_from_directory,
    abort, jsonify, Response, redirect, url_for
)

from tornado.wsgi import WSGIContainer
from tornado.web import Application, FallbackHandler
from tornado.websocket import WebSocketHandler
from tornado.ioloop import IOLoop

from progressivis.core.scheduler import Scheduler

class WebSocket(WebSocketHandler):
    def open(self):
        print("Socket opened.")

    def on_message(self, message):
        self.write_message("Received: " + message)
        print("Received message: " + message)

def on_close(self):
    print("Socket closed.")


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


def app_create(config="settings.py", scheduler=None):
    if scheduler is None:
        scheduler = Scheduler.default
    app = Flask('progressivis.server')
    if isinstance(config, str):
        app.config.from_pyfile(config)
    elif isinstance(config, dict):
        app.config.update(config)

    app.register_blueprint(progressivis_bp)
    progressivis_bp.setup(scheduler)

    container = WSGIContainer(app)
    server = Application([
        (r'/websocket/', WebSocket),
        (r'.*', FallbackHandler, dict(fallback=container))
    ])
    server.listen(5000)
    return server

def app_run(app):
    IOLoop.instance().start()
