"""

"""
import time

from progressivis import (
    Scheduler,
    CSVLoader,
)
from progressivis.datasets import get_dataset
from progressivis.stats.kernel_density import KernelDensity
import numpy as np
sampleN = 30
samples = np.indices((sampleN + 1, sampleN + 1)).reshape(2, -1).T / sampleN * 3 - 1.5


def _print_len(x):
    if x is not None:
        print(len(x))

#log_level() #package='progressivis.stats.histogram2d')

try:
    s = scheduler
except NameError:
    s = Scheduler()

CSV = CSVLoader(get_dataset('bigfile_mvn'), index_col=False, header=None, scheduler=s)

#PR = Every(scheduler=s)
#PR.input.df = CSV.output.table
KNNKDE = KernelDensity(scheduler=s, samples=samples, bins=sampleN)
KNNKDE.input.table = CSV.output.table
if __name__ == '__main__':
    s.start()
    while True:
        time.sleep(2)
        s.to_json()
        KNNKDE.to_json()  # simulate a web query
        # SCATTERPLOT.get_image()
    s.join()
    print(len(CSV.table()))
