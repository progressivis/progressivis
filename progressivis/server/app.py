"""
Flask server for ProgressiVis.
"""
from __future__ import absolute_import, division, print_function

import logging

from six import StringIO

import numpy as np

from flask import Flask, Blueprint
from flask.json import JSONEncoder
from tornado.wsgi import WSGIContainer
from tornado.web import Application, FallbackHandler
from tornado.websocket import WebSocketHandler
from tornado.ioloop import IOLoop
from progressivis import ProgressiveError
from progressivis.core.scheduler import Scheduler



logger = logging.getLogger(__name__)

class JSONEncoder4Numpy(JSONEncoder):
    "Encode numpy objects"
    def default(self, obj):
        "Handle default encoding."
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class ProgressiveWebSocket(WebSocketHandler):
    "Manage the WebSocket connection"
    sockets_for_path = {}
    def __init__(self, application, request, **kwargs):
        super(ProgressiveWebSocket, self).__init__(application, request, **kwargs)
        self.handshake = False
        self.path = None

    def data_received(self, chunk):
        pass

    def register(self):
        "Register the websocket to ProgressiVis"
        if self.path is None:
            logger.error('Trying to register a websocket without path')
            raise ProgressiveError('Trying to register a websocket without path')
        if self.path not in self.sockets_for_path:
            self.sockets_for_path[self.path] = [self]
        else:
            self.sockets_for_path[self.path].append(self)

    def unregister(self):
        "Unregister the websocket from ProgressiVis"
        if self.path is None or self.path not in self.sockets_for_path:
            return
        self.sockets_for_path[self.path].remove(self)

    def open(self, *args, **kwargs):
        "Open the connection"
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
        logger.warning("Received message '%s' from '%s'", message, self.path)

    def on_close(self):
        self.unregister()
        logger.info("Socket for '%s' closed.", self.path)

    def select_subprotocol(self, subprotocols):
        # for now, don't implement any subprotocol
        return None

    @staticmethod
    def write_to_path(path, msg):
        "Write message to web page monitoring the specific path"
        logger.info('Sending message %s to path %s', msg, path)
        sockets = ProgressiveWebSocket.sockets_for_path.get(path)
        if not sockets:
            logger.warning('Message sent to nonexistent path "%s"', path)
            return
        for s in sockets:
            s.write_message(msg)


class ProgressivisBlueprint(Blueprint):
    "Blueprint for ProgressiVis"
    def __init__(self, *args, **kwargs):
        super(ProgressivisBlueprint, self).__init__(*args, **kwargs)
        self.scheduler = None
        self.start_logging()

    def start_logging(self):
        "Start logging out"
        out = self._log_stream = StringIO()
        out.write("<html><body><table>"
                  "<tr><th>Time</th><th>Name</th><th>Level</th><th>Message</th></tr>\n")
        streamhandler = logging.StreamHandler(stream=self._log_stream)
        streamhandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('<tr><td>%(asctime)s</td>'
                                      '<td>%(name)s</td>'
                                      '<td>%(levelname)s</td>'
                                      '<td>%(message)s</td></tr>\n')
        streamhandler.setFormatter(formatter)
        logging.getLogger("progressivis").addHandler(streamhandler)
        logging.getLogger("progressivis").setLevel(logging.DEBUG)

    def setup(self, scheduler):
        "Setup the connection with the scheduler"
        self.scheduler = scheduler
        self.scheduler._tick_proc = self.tick_scheduler

    def tick_scheduler(self, scheduler, run_number):
        "Run at each tick"
        # pylint: disable=unused-argument, no-self-use
        ProgressiveWebSocket.write_to_path('scheduler', 'tick %d'%run_number)

    def step_tick_scheduler(self, scheduler, run_number):
        "Run at each step"
        # pylint: disable=no-self-use
        ProgressiveWebSocket.write_to_path('scheduler', 'tick %d'%run_number)
        scheduler.stop()

    def step_once(self):
        "Run once"
        self.scheduler.start(tick_proc=self.step_tick_scheduler) # i.e. step+write_to_path

    def start(self):
        "Run when the scheduler starts"
        self.scheduler.start(tick_proc=self.tick_scheduler)

    def tick_module(self, module, run_number):
        "Run when a module has run"
        # pylint: disable=no-self-use
        ProgressiveWebSocket.write_to_path('module %s'%module.id, 'tick %d'%run_number)

    def get_log(self):
        "Return the log"
        self._log_stream.flush()
        return self._log_stream.getvalue().replace("\n", '<br>')


progressivis_bp = ProgressivisBlueprint('progressivis.server',
                                        'progressivis.server',
                                        static_folder='static',
                                        static_url_path='/progressivis/static',
                                        template_folder='templates')


def app_create(config="settings.py", scheduler=None):
    "Create the application"
    if scheduler is None:
        scheduler = Scheduler.default
    app = Flask('progressivis.server')
    if isinstance(config, str):
        app.config.from_pyfile(config)
    elif isinstance(config, dict):
        app.config.update(config)
    app.json_encoder = JSONEncoder4Numpy
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
    "Run the application"
    # pylint: disable=unused-argument
    IOLoop.instance().start()

def start_server(scheduler=None, debug=False):
    "Start the server"
    if scheduler is None:
        scheduler = Scheduler.default
    print('Scheduler has %d modules' % len(scheduler))
    app = app_create(scheduler)
    app.debug = debug
    app_run(app)
