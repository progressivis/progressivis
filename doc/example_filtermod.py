from progressivis import Sink, Scheduler, RandomPTable
from progressivis.table.filtermod import FilterMod

scheduler = Scheduler()
with scheduler:
    random = RandomPTable(2, rows=100000, scheduler=scheduler)
    filter_ = FilterMod(scheduler=scheduler)
    filter_.params.expr = "_1 > 0.5"
    filter_.input.table = random.output.result
    sink = Sink(scheduler=scheduler)
    sink.input.inp = filter_.output.result
