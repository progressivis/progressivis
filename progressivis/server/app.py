from __future__ import print_function

import logging
logger = logging.getLogger(__name__)

from flask import Flask, Blueprint

from tornado.wsgi import WSGIContainer
from tornado.web import Application, FallbackHandler
from tornado.websocket import WebSocketHandler
from tornado.ioloop import IOLoop

from progressivis import ProgressiveError, Scheduler


class ProgressiveWebSocket(WebSocketHandler):
    sockets_for_path = {}
    def __init__(self, application, request, **kwargs):
        super(ProgressiveWebSocket, self).__init__(application, request, **kwargs)
        self.handshake = False
        self.path = None

    def register(self):
        if self.path is None:
            logger.error('Trying to register a websocket without path')
            raise ProgressiveError('Trying to register a websocket without path')
        if self.path not in self.sockets_for_path:
            self.sockets_for_path[self.path] = [self]
        else:
            self.sockets_for_path[self.path].append(self)

    def unregister(self):
        if self.path is None or self.path not in self.sockets_for_path:
            return
        self.sockets_for_path[self.path].remove(self)

    def open(self):
        logger.info("Socket opened.")
        self.handshake = False

    def on_message(self, message):
        #self.write_message("Received: " + message)
        if not self.handshake:
            if message.startswith('ping '):
                self.path = message[5:]
                self.register()
                self.write_message('pong')
                self.handshake = True
                logger.debug("Handshake received for path: '%s'", self.path)
            else:
                logger.error("Received msg before handshake: '%s'", message)
            return
        logger.warn("Received message '%s' from '%s'", message, self.path)

    def on_close(self):
        self.unregister()
        logger.info("Socket for '%s' closed.", self.path)

    def select_subprotocol(self, subprotocols):
        # for now, don't implement any subprotocol
        return None

    @staticmethod
    def write_to_path(path, msg):
        sockets = ProgressiveWebSocket.sockets_for_path.get(path)
        if not sockets:
            return
        for s in sockets:
            s.write_message(msg)


class ProgressivisBlueprint(Blueprint):
    def __init__(self, *args, **kwargs):
        super(ProgressivisBlueprint, self).__init__(*args, **kwargs)

    def setup(self, scheduler):
        self.scheduler = scheduler
        self.scheduler._tick_proc = self.tick_scheduler

    def tick_scheduler(self, scheduler, run_number):
        ProgressiveWebSocket.write_to_path('scheduler', 'tick %d'%run_number)

    def tick_module(self, module, run_number):
        ProgressiveWebSocket.write_to_path('module %s'%module.id, 'tick %d'%run_number)



progressivis_bp = ProgressivisBlueprint('progressivis.server',
                                        'progressivis.server',
                                        static_folder='static',
                                        static_url_path='/progressivis/static',
                                        template_folder='templates')

import views
views # to shut up pyflakes

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
        (r'/websocket/', ProgressiveWebSocket),
        (r'.*', FallbackHandler, dict(fallback=container))
    ])
    server.listen(5000)
    return server

def app_run(app):
    IOLoop.instance().start()
