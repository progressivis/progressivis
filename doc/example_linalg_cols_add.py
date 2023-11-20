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
        cols_out=["x", "y", "z"],
        scheduler=scheduler,
    )
    module.input.table = random.output.result[["_3", "_5", "_7"], ["_4", "_6", "_8"]]
    sink = Sink(scheduler=scheduler)
    sink.input.inp = module.output.result
