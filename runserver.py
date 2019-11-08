"""
Starts the progressivis server and launches a python progressivis application.
"""
import sys
import signal
import logging
from os import getenv
import asyncio as aio
import requests

from progressivis import log_level
from progressivis.server import start_server
from progressivis.core.scheduler import Scheduler

ENV = {}

for filename in sys.argv[1:]:
    if filename == "nosetests":
        continue
    print("Loading '%s'" % filename)
    # pylint: disable=exec-used
    ENV['scheduler'] = Scheduler.default
    exec(compile(open(filename).read(), filename, 'exec'), ENV, ENV)

def _signal_handler(signum, frame):
    # pylint: disable=unused-argument
    requests.get('http://localhost:5000/exit')
    sys.exit()

if __name__ == '__main__':
    log_level(level=logging.NOTSET)
    #signal.signal(signal.SIGTERM, _signal_handler)
    #signal.signal(signal.SIGINT, _signal_handler)
    aio.run(start_server())
    print("Web Server launched!")
    #while True:
    #    _ = input()
