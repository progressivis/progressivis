"""Multi-Thread Scheduler, meant to run in its own thread."""

import logging
from .scheduler_base import BaseScheduler
from progressivis.utils.errors import ProgressiveError


logger = logging.getLogger(__name__)


class Scheduler(BaseScheduler):
    """
    Main scheduler class.

    Manage the execution of the progressive workflow in its own thread.
    """
    def __init__(self, interaction_latency=1):
        super(Scheduler, self).__init__(interaction_latency)
        self.thread = None
        self.thread_name = "Progressive Scheduler"


    @staticmethod
    def set_default():
        "Set the default scheduler."
        if not isinstance(BaseScheduler.default, Scheduler):
            BaseScheduler.default = Scheduler()

    def _before_run(self):
        logger.debug("Before run %d", self._run_number)

    async def start(self, tick_proc=None, idle_proc=None, coros=()):
        self.coros=list(coros)
        if tick_proc:
            assert callable(tick_proc)
            self._tick_procs = [tick_proc]
        else:
            self._tick_procs = []
        if idle_proc:
            assert callable(idle_proc)
            self._idle_procs = [idle_proc]
        else:
            self._idle_procs = []
        await self.run()

    def _after_run(self):
        logger.debug("After run %d", self._run_number)

    def done(self):
        self.thread = None
        logger.info('Thread finished')


if BaseScheduler.default is None:
    BaseScheduler.default = Scheduler()
