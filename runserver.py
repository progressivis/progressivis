from progressivis.server.app import start_server
from progressivis.core.scheduler import Scheduler

from progressivis.core.utils import Thread
import sys
import signal
from six.moves import input
import requests


env = {'scheduler': Scheduler.default }

for fn in sys.argv[1:]:
    if fn=="nosetests":
        continue
    print("Loading '%s'" % fn)
    exec(compile(open(fn).read(), fn, 'exec'), env, env)

def signal_handler(signum, frame):
    requests.get('http://localhost:5000/exit')
    sys.exit()
    
if __name__=='__main__':
    #log_level()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)    
    th = Thread(target=start_server)
    th.start()
    print("Server launched!")
    while True:
        _ = input()

    
    
