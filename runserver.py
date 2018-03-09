"""
Starts the progressivis server and launches a python progressivis application.
"""
import sys
import signal
import logging

from six.moves import input
import requests

from progressivis import log_level
from progressivis.server.app import start_server
from progressivis.core.scheduler import Scheduler

from progressivis.core.utils import Thread


ENV = {'scheduler': Scheduler.default}

for filename in sys.argv[1:]:
    if filename == "nosetests":
        continue
    print("Loading '%s'" % filename)
    exec(compile(open(filename).read(), filename, 'exec'), ENV, ENV)

def _signal_handler(signum, frame):
    # pylint: disable=unused-argument
    requests.get('http://localhost:5000/exit')
    sys.exit()

if __name__ == '__main__':
    log_level(level=logging.NOTSET)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    thread = Thread(target=start_server)
    thread.start()
    print("Server launched!")
    while True:
        _ = input()
