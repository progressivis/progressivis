from progressivis import Scheduler, Print, RandomTable
from progressivis.vis import MCScatterPlot
from progressivis.cluster import MBKMeans
import asyncio as aio

s = Scheduler()

table = RandomTable(columns=['a', 'b'], rows=50000, throttle=500, scheduler=s)
mbkmeans = MBKMeans(columns=['a', 'b'], n_clusters=8, batch_size=100, is_input=False, scheduler=s)
mbkmeans.input.table = table.output.table
prn = Print(scheduler=s)
prn.input.df = mbkmeans.output.table
sp = MCScatterPlot(scheduler=s, classes=[('Scatterplot', 'a', 'b')], approximate=True)
sp.create_dependent_modules(mbkmeans,'table')
sp['Scatterplot'].range_query_2d.hist_index_x.params.init_threshold = 1
sp['Scatterplot'].range_query_2d.hist_index_y.params.init_threshold = 1

if __name__ == '__main__':
    #table.start()
    aio.run(s.start(coros=[aio.sleep(3600)]))
