"Test the Histograms visualization module"
from progressivis import Scheduler, Every
from progressivis.stats import RandomTable, Min, Max
from progressivis.vis import Histograms

import numpy as np

#log_level()

try:
    s = scheduler
    print('No scheduler defined, using the standard one')
except NameError:
    s = Scheduler()

def main():
    "Main function"
    csvmod = RandomTable(columns=['a', 'b', 'c'], rows=1000000,
                         random=np.random.randn,
                         throttle=1000, scheduler=s)
    minmod = Min(scheduler=s)
    minmod.input.table = csvmod.output.table
    maxmod = Max(scheduler=s)
    maxmod.input.table = csvmod.output.table
    histograms = Histograms(scheduler=s)
    histograms.input.table = csvmod.output.table
    histograms.input.min = minmod.output.table
    histograms.input.max = maxmod.output.table
    prlen = Every(scheduler=s)
    prlen.input.df = histograms.output.table
    return csvmod

main()

if __name__ == '__main__':
    print("Starting")
    main().start()
