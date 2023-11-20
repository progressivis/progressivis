from progressivis.core import Sink, Scheduler
from progressivis.stats import RandomPTable
from progressivis.linalg import Add

#
# every column in random1 are added to the column in random2 in the same position
#

scheduler = Scheduler()
with scheduler:
    random1 = RandomPTable(3, rows=100_000, scheduler=scheduler)
    random2 = RandomPTable(3, rows=100_000, scheduler=scheduler)
    module = Add(scheduler=scheduler)
    module.input.first = random1.output.result
    module.input.second = random2.output.result
    sink = Sink(scheduler=scheduler)
    sink.input.inp = module.output.result

#
# columns _3, _5, _7 in random1 are added to column _4, _6, _8 in random2
#

scheduler = Scheduler()
with scheduler:
    random1 = RandomPTable(3, rows=100_000, scheduler=scheduler)
    random2 = RandomPTable(3, rows=100_000, scheduler=scheduler)
    module = Add(scheduler=scheduler)
    module.input.first = random1.output.result["_3", "_5", "_7"]
    module.input.second = random2.output.result["_4", "_6", "_8"]
    sink = Sink(scheduler=scheduler)
    sink.input.inp = module.output.result
