from progressivis import Scheduler, Print
from progressivis.cluster import MBKMeans
from progressivis.stats import RandomTable
from progressivis.vis import ScatterPlot

try:
    s = scheduler
except:
    s = Scheduler()

table = RandomTable(columns=['a', 'b'], rows=50000, throttle=500, scheduler=s)
mbkmeans = MBKMeans(columns=['a', 'b'], n_clusters=8, batch_size=100, is_input=False, scheduler=s)
mbkmeans.input.df = table.output.df
prn = Print(scheduler=s)
prn.input.df = mbkmeans.output.df
#sp = ScatterPlot('a', 'b')
#sp.create_dependent_modules(mbkmeans,'df')

if __name__ == '__main__':
    table.start()
