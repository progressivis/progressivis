from progressivis.server.app import create_app
from progressivis.core.scheduler import Scheduler

import sys

env = {'scheduler': Scheduler.default }

for fn in sys.argv[1:]:
    print "Loading '%s'" % fn
    execfile(fn, env, env)

if __name__=='__main__':
    app = create_app()
    app.run(debug=True)
