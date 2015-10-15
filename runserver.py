from progressivis import log_level
from progressivis.server.app import app_create, app_run
from progressivis.core.scheduler import Scheduler
from progressivis.core.mt_scheduler import MTScheduler

import sys

MTScheduler.set_default()


env = {'scheduler': Scheduler.default }

for fn in sys.argv[1:]:
    print "Loading '%s'" % fn
    execfile(fn, env, env)

if __name__=='__main__':
    #log_level()
    print 'Scheduler has %d modules' % len(Scheduler.default)
    app = app_create(Scheduler.default)
    app.debug = True
    
    app_run(app)
