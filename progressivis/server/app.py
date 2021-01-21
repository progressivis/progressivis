"""
AIOHTTP server for ProgressiVis.

The server provides the main user interface for progressivis.
It serves pages related to the scheduler and modules.
For all the modules, it provides access to their internal state and
to the tables it maintains.

When the scheduler is running, the server implements a simple protocol.
Each web page shown on a browser also opens a socketio connection.
The server sends one message "tick" throught the socketio when the served entity is been changed.
When the client/browser receives it, it sends a request to get the data from the module. This
request is made through the socketio directly, and the value is returned, allowing the next tick
to be sent by the server. This mechansim is meant to get a responsive browser with asynchronous
updates.  Between the time the "tick" is received by the browser and the value is sent back by
the server, many iterations may have been run.  The browser receives data as fast as it can, and
the server sends a simple notification and serves the data as fast as it can.
"""
import time
import logging
import logging.config
from functools import partial
from io import StringIO

import numpy as np

#from flask import Flask, Blueprint, request, json as flask_json
#from flask.json import JSONEncoder
#from flask_socketio import SocketIO, join_room, send
#import eventlet
import json as js
import aiohttp
from aiohttp import web
import socketio as sio
import pathlib
import jinja2
import aiohttp_jinja2
#import aiohttp_debugtoolbar
#from aiohttp_debugtoolbar import toolbar_middleware_factory

import progressivis.core.aio as aio
from progressivis import Scheduler, Module
from ..core import JSONEncoderNp

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name
socketio = None


class ProgressivisBlueprint(web.Application):
    "Blueprint for ProgressiVis"
    def __init__(self, *args, **kwargs):
        super(ProgressivisBlueprint, self).__init__(*args, **kwargs)
        self._sids_for_path = {}
        self._run_number_for_sid = {}
        self.scheduler = None
        self.hotline_set = set()
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
        self.scheduler.on_tick(self.tick_scheduler)

    def register_module(self, path, sid):
        "Register a module with a specified path"
        if sid in self._run_number_for_sid:
            self._run_number_for_sid[sid] = 0
            return
        print('Register module:', path, 'sid:', sid)
        self._run_number_for_sid[sid] = 0
        if path in self._sids_for_path:
            sids = self._sids_for_path[path]
            sids.add(sid)
        else:
            self._sids_for_path[path] = set([sid])

    def unregister_module(self, sid):
        "Unregister a specified path"
        if sid in self._run_number_for_sid:
            del self._run_number_for_sid[sid]
        for sids in self._sids_for_path.values():
            if sid in sids:
                sids.remove(sid)
                return

    def sids_for_path(self, path):
        "Get the sid list from a path"
        return self._sids_for_path.get(path, set())

    def sid_run_number(self, sid):
        "Return the last run_number sent for the specified sid"
        return self._run_number_for_sid.get(sid, 0)

    def _prevent_tick(self, sid, run_number, ack):
        if ack:
            self._run_number_for_sid[sid] = run_number
        else:
            logging.debug('Ack not well received')
            print('Preventing ticks for', sid)

    def reset_sid(self, sid):
        "Resets the sid to 0 to stop sending ticks for that sid"
        #print('Reseting sid', sid)
        self._run_number_for_sid[sid] = 0

    async def emit_tick(self, path, run_number, payload=None):
        "Emit a tick unless it has not been acknowledged"
        sids = self.sids_for_path(path)
        for sid in set(sids): # size could change
            if self._run_number_for_sid[sid] == 0:
                #print('Emiting tick for', sid, 'in path', path)
                json_ = {'run_number': run_number}
                if payload is not None: json_['payload'] = payload
                await socketio.emit('tick', json_, room=sid,
                              callback=partial(self._prevent_tick, sid, run_number))
            #else:
            #    #print('No tick for', sid, 'in path', path)
        #await aio.sleep(0) # yield ...

    async def tick_scheduler(self, scheduler, run_number):
        "Run at each tick"
        # pylint: disable=unused-argument
        await self.emit_tick('scheduler', run_number)

    async def step_tick_scheduler(self, scheduler, run_number):
        "Run at each step"
        await scheduler.stop()
        await self.emit_tick('scheduler', run_number)

    def step_once(self):
        "Run once"
        #self.scheduler.resume(tick_proc=self.step_tick_scheduler) # i.e. step+write_to_path
        self.scheduler.task_start(tick_proc=self.step_tick_scheduler) # i.e. step+write_to_path

    def start(self):
        "Run when the scheduler starts"
        self.scheduler.task_start(tick_proc=self.tick_scheduler)
        
    def tick_module(self, module, run_number):
        "Run when a module has run"
        # pylint: disable=no-self-use
        payload = None
        if module.name in self.hotline_set:
            payload = module.to_json()
        aio.create_task(self.emit_tick(module.name, run_number, payload=payload))

    def get_log(self):
        "Return the log"
        self._log_stream.flush()
        return self._log_stream.getvalue().replace("\n", '<br>')

PROJECT_ROOT = pathlib.Path(__file__).parent

progressivis_bp = ProgressivisBlueprint() #middlewares=[toolbar_middleware_factory])
#aiohttp_debugtoolbar.setup(progressivis_bp)
progressivis_bp.router.add_static('/static/',
                          path=PROJECT_ROOT / 'static',
                          name='static')
aiohttp_jinja2.setup(progressivis_bp,
                     loader=jinja2.FileSystemLoader(str(PROJECT_ROOT / 'templates')))

def path_to_module(path):
    """
    Return a module given its path, or None if not found.
    A path is the module id alone, or the toplevel module module id
    followed by modules stored inside it.

    For example 'scatterplot/range_query' will return the range_query
    module used as dependent module of scatterplot.
    """
    scheduler = progressivis_bp.scheduler
    #print('module_path(%s)'%(path))
    ids = path.split('/')
    module = scheduler.modules()[ids[0]]
    if module is None:
        return None
    for subid in ids[1:]:
        if not hasattr(module, subid):
            return None
        module = getattr(module, subid)
        if not isinstance(module, Module):
            return None
    return module

#@on.socketio('join')
def _on_join(sid, json):
    if json.get("type") != "ping":
        logging.error("Expected a ping message")
    path = json["path"]
    print('socketio join received for "%s"'% path)
    socketio.enter_room(sid, path)
    #print('socketio Roomlist:', rooms())
    return {'type': 'pong'}

##@on.socketio('connect')
def _on_connect(sid, _):
    print('socketio connect ', sid)

#@on.socketio('disconnect')
def _on_disconnect(sid):
    progressivis_bp.unregister_module(sid)
    print('socketio disconnect ', sid)

#@on.socketio('/progressivis/scheduler/start')
def _on_start(sid):
    scheduler = progressivis_bp.scheduler
    #if scheduler.is_running():
    if not scheduler.is_stopped():
        return {'status': 'failed',
                'reason': 'scheduler is already running'}
    progressivis_bp.start()
    return {'status': 'success'}

#@on.socketio('/progressivis/scheduler/stop')
def _on_stop(sid):
    scheduler = progressivis_bp.scheduler
    #if not scheduler.is_running():
    if scheduler.is_stopped():    
        return {'status': 'failed',
                'reason': 'scheduler is not is_running'}
    scheduler.task_stop()
    return {'status': 'success'}

#@on.socketio('/progressivis/scheduler/step')
def _on_step(sid):
    scheduler = progressivis_bp.scheduler
    #if scheduler.is_running():
    if not scheduler.is_stopped():    
        return {'status': 'failed',
              'reason': 'scheduler is is_running'}
    progressivis_bp.step_once()
    return {'status': 'success'}

#@on.socketio('/progressivis/scheduler')
def _on_scheduler(sid, short=False):
    scheduler = progressivis_bp.scheduler
    #print('socketio scheduler called')
    progressivis_bp.register_module('scheduler', sid)
    #print(progressivis_bp._sids_for_path)
    assert sid in progressivis_bp.sids_for_path('scheduler')
    return scheduler.to_json(short)

#@on.socketio('/progressivis/module/get')
def _on_module_get(sid, path, *unused_all, **kwargs ):
    module = path_to_module(path)
    print("calling  _on_module_get")
    if module is None:
        return {'status': 'failed',
                'reason': 'unknown module %s'%path}
    progressivis_bp.register_module(module.name, sid)
    module.set_end_run(progressivis_bp.tick_module) # setting it multiple time is ok
    #print('on_module_get', path)
    return module.to_json()

#@on.socketio('/progressivis/module/hotline_on')
def _on_module_hotline_on(sid, path):
    module = path_to_module(path)
    if module is None:
        return {'status': 'failed',
                'reason': 'unknown module %s'%path}
    progressivis_bp.hotline_set.add(module.name)

#@on.socketio('/progressivis/module/hotline_off')
def _on_module_hotline_off(sid, path):
    module = path_to_module(path)
    if module is None:
        return {'status': 'failed',
                'reason': 'unknown module %s'%path}
    try:
        progressivis_bp.hotline_set.remove(module.name)
    except:
        pass

#@on.socketio('/progressivis/module/input')
async def _on_module_input(sid, data):
    data = js.loads(data)
    path = None
    var_values = None
    try:
        path = data['path']
        var_values = data['var_values']
    except KeyError:
        pass
    module = path_to_module(path)
    if module is None:
        return {'status': 'failed',
                'reason': 'unknown module %s'%path}
    if var_values is None:
        return {'status': 'failed',
                'reason': 'no var_values for %s'%path}
    try:
        print('sending to %s: %s'%(module.name, var_values))
        msg = await module.from_input(var_values)
        # pylint: disable=broad-except
    except Exception as exc:
        msg = str(exc)
        print('Error: %s'%msg)
        return {'status': 'failed', 'reason': 'Cannot input: %s' % msg}

    print('success: %s'%msg)
    ret = {'status': 'success'}
    if msg:
        ret['error'] = msg
    return ret

#@on.socketio('/progressivis/module/df')
def _on_module_df(sid, path):
    (mid, slot) = path.split('/')
    #print('socketio Getting module', mid, 'slot "'+slot+'"')
    module = path_to_module(mid)
    if module is None:
        return {'status': 'failed',
                'reason': 'invalid module'}
    df = module.get_data(slot)
    if df is None:
        return {'status': 'failed',
                'reason': 'invalid data'}
    return {'columns':['index']+df.columns}

#@on.socketio('/progressivis/module/quality')
def _on_module_quality(sid, mid):
    #print('socketio quality for', mid)
    module = path_to_module(mid)
    if module is None:
        return {'status': 'failed',
                'reason': 'invalid module'}
    slot = '_trace'
    #print('socketio Getting slot "'+slot+'"')
    df = module.get_data(slot)
    if df is None:
        return {'status': 'failed',
                'reason': 'invalid data'}
    qual = df['quality'].values[np.nonzero(df['quality'].values>0)]
    return {'index':df.index, 'quality': qual}

#@on.socketio('/progressivis/logger')
def _on_logger(sid):
    managers = logging.Logger.manager.loggerDict
    ret = []
    for (module, log) in managers.items():
        if isinstance(log, logging.Logger):
            ret.append({'module': module,
                        'level': logging.getLevelName(log.getEffectiveLevel())})
    def _key_log(a):
        return a['module'].lower()
    ret.sort(key=_key_log)
    return {'loggers': ret}


def app_create(config="settings.py", scheduler=None):
    "Create the application"
    if scheduler is None:
        scheduler = Scheduler.default
    app = web.Application() #middlewares=[toolbar_middleware_factory])
    #aiohttp_debugtoolbar.setup(app)
    if isinstance(config, str):
        #app['config'] = get_config(config)
        pass
    elif isinstance(config, dict):
        app['config'].update(config)
    app.add_subapp('/progressivis/', progressivis_bp)
    progressivis_bp.setup(scheduler)
    return app
# https://stackoverflow.com/questions/7507825/where-is-a-complete-example-of-logging-config-dictconfig
LOGGING_CONFIG = { 
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
    },
    'loggers': { 
        '': {  # root logger
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
        'aiohttp.access': { 
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'aiohttp.client': { 
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'aiohttp.internal': { 
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'aiohttp.server': { 
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'aiohttp.web': { 
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'aiohttp.websocket': { 
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        '__main__': {  # if __name__ == '__main__'
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
    } 
}
#logging.config.dictConfig(LOGGING_CONFIG)

class _AsyncServer(sio.AsyncServer):
    def on_event(self, message, handler, namespace=None):
        self.on(message, namespace=namespace)(handler)

async def _inf_loop(n):
    while True:
        await aio.sleep(n)
        print(":")
        
async def start_server(scheduler=None, debug=False):
    "Start the server"
    # pylint: disable=global-statement
    global socketio
    #eventlet.monkey_patch() # see https://github.com/miguelgrinberg/Flask-SocketIO/issues/357
    if scheduler is None:
        scheduler = Scheduler.default
    print('Scheduler %s has %d modules' % (scheduler.__class__.__name__, len(scheduler)))
    app = app_create(scheduler)
    app['debug'] = debug
    #socketio = _AsyncServer(async_mode='aiohttp')
    socketio = _AsyncServer(async_mode='aiohttp', json=JSONEncoderNp)
    socketio.on_event('connect', _on_connect)
    socketio.on_event('disconnect', _on_disconnect)
    socketio.on_event('join', _on_join)
    socketio.on_event('/progressivis/scheduler/start', _on_start)
    socketio.on_event('/progressivis/scheduler/step', _on_step)
    socketio.on_event('/progressivis/scheduler/stop', _on_stop)
    socketio.on_event('/progressivis/scheduler', _on_scheduler)
    socketio.on_event('/progressivis/module/get', _on_module_get)
    socketio.on_event('/progressivis/module/hotline_on', _on_module_hotline_on)
    socketio.on_event('/progressivis/module/hotline_off', _on_module_hotline_off)
    socketio.on_event('/progressivis/module/df', _on_module_df)
    socketio.on_event('/progressivis/module/input', _on_module_input)
    socketio.on_event('/progressivis/module/quality', _on_module_quality)
    socketio.on_event('/progressivis/logger', _on_logger)
    socketio.attach(app)
    #socketio.run(app)
    #web.run_app(app)
    runner = web.AppRunner(app) #, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    print('Server started, connect to http://localhost:8080/progressivis/scheduler.html')
    srv =  site.start()
    #await scheduler.start(tick_proc=progressivis_bp.tick_scheduler, coros=[srv, aio.sleep(3600)])
    sch_task = scheduler.start(tick_proc=progressivis_bp.tick_scheduler)
    await aio.gather(srv, sch_task, _inf_loop(3600))
    #await aio.sleep(3600)
    #logging.basicConfig(level=logging.DEBUG)
def stop_server():
    "Stop the server"
    socketio.stop()
