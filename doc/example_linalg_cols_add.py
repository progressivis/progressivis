from progressivis.core import Sink, Scheduler
from progressivis.stats import RandomPTable
from progressivis.linalg import ColsAdd

#
# columns _3, _5, _7 are added to column _4, _6, _8
#

scheduler = Scheduler()
with scheduler:
    random = RandomPTable(3, rows=100_000, scheduler=scheduler)
    module = ColsAdd(
        columns={"first": ["_3", "_5", "_7"], "second": ["_4", "_6", "_8"]},
        scheduler=scheduler,
    )
    module.input.table = random.output.result
    sink = Sink(scheduler=scheduler)
    sink.input.inp = module.output.result
