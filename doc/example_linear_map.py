from progressivis import Sink, Scheduler, RandomPTable
from progressivis.linalg.linear_map import LinearMap

scheduler = Scheduler()

with scheduler:
    vectors = RandomPTable(20, rows=100000, scheduler=scheduler)
    transf = RandomPTable(20, rows=3, scheduler=scheduler)
    module = LinearMap(scheduler=scheduler)
    module.input.vectors = vectors.output.result["_3", "_4", "_5"]
    module.input.transformation = transf.output.result["_4", "_5", "_6", "_7"]
    sink = Sink(scheduler=scheduler)
    sink.input.inp = module.output.result
