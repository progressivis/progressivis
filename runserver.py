"""
Starts the progressivis server and launches a python progressivis application.
"""
import sys
import signal
import logging
from os import getenv

from six.moves import input
import requests

from progressivis import log_level
from progressivis.server import start_server
from progressivis.core.scheduler import Scheduler, BaseScheduler

from progressivis.core.utils import Thread


ENV = {'scheduler': Scheduler.default}

for filename in sys.argv[1:]:
    if filename == "nosetests":
        continue
    if getenv("NOTHREAD"):
        ENV['scheduler'] = BaseScheduler()
    print("Loading '%s'" % filename)
    # pylint: disable=exec-used
    exec(compile(open(filename).read(), filename, 'exec'), ENV, ENV)

def _signal_handler(signum, frame):
    # pylint: disable=unused-argument
    requests.get('http://localhost:5000/exit')
    sys.exit()

if __name__ == '__main__':
    log_level(level=logging.NOTSET)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    THREAD = Thread(target=start_server)
    THREAD.start()
    print("Server launched!")
    while True:
        _ = input()
