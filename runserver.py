"""
Starts the progressivis server and launches a python progressivis application.
"""
import sys
import logging
import asyncio as aio

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


if __name__ == '__main__':
    log_level(level=logging.NOTSET)
    aio.run(start_server())
