from progressivis import Sink, Scheduler, RandomPTable, RangeQuery


scheduler = Scheduler()
with scheduler:
    random = RandomPTable(2, rows=100000, scheduler=scheduler)
    range_qry = RangeQuery(column="_1", scheduler=scheduler)
    range_qry.create_dependent_modules(random, "result")
    sink = Sink(scheduler=scheduler)
    sink.input.inp = range_qry.output.result
    sink.input.inp = range_qry.output.min
    sink.input.inp = range_qry.output.max
