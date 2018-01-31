from __future__ import absolute_import, division, print_function

import time
import logging
from progressivis.core.module import Module

logger = logging.getLogger(__name__)

__all__ = ['Sentinel']


class Sentinel(Module):
    """
    Module that is run by the scheduler to sleep a bit
    if nothing else is running.
    """
    def __init__(self, min_time=0.2, **kwds):
        super(Sentinel, self).__init__(**kwds)
        self._min_time = min_time

    def predict_step_size(self, duration):
        return 1

    def run_step(self, run_number, step_size, howlong):
        logger.debug('last: %s, current: %s', self.last_update(), run_number)
        if self.scheduler().run_queue_length() == 0:  # lonely in the queue
            return self._return_run_step(Module.state_zombie, steps_run=1)
        if self.last_update() is (run_number-1):  # nothing else was run
            idle = self.scheduler().idle_proc()
            if idle is None:
                logger.info('sleeping %f', self._min_time)
                time.sleep(self._min_time)
            else:
                idle(self.scheduler(), run_number)
        return self._return_run_step(Module.state_blocked, steps_run=1)
