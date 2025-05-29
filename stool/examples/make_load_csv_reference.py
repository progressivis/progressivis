import pandas as pd
from progressivis import Scheduler
from progressivis.io import CSVLoader
import subprocess
import glob
from progressivis.core.utils import RandomBytesIO
from stool import BenchmarkCase, bench
import unittest

GIGA = 1000000000

class BenchLoadCsv(BenchmarkCase):
    def __init__(self, testMethod='runTest'):
        super(BenchLoadCsv, self).__init__(testMethod)
        self.nb_step = 8
        self.random_file = None
    def tearDown(self):
        for d in glob.glob('/tmp/progressivis_*'):
            subprocess.call(['/bin/rm','-rf', d])
    def setUp(self):
        self.set_step_header("Gigabytes")
    def setUpStep(self, step):
        self.set_step_info("{} Gb".format(step))
    def tearDownStep(self, step):
        pass
    @bench(name="Nop")
    def none_read_csv(self):
        for _ in RandomBytesIO(cols=30, size=self.current_step*GIGA):
            pass
    @bench(name="Progressivis", corrected_by="Nop")
    def p10s_read_csv(self):
        s=Scheduler()
        module=CSVLoader(RandomBytesIO(cols=30, size=self.current_step*GIGA), header=None, scheduler=s)
        module.start()

    @bench(name="Pandas", corrected_by="Nop")
    def pandas_read_csv(self):
        pd.read_csv(RandomBytesIO(cols=30, size=self.current_step*GIGA))
    #@bench(name="Dask", corrected_by="Nop")
    #def dask_read_csv(self):
    #    dd.read_csv(RandomBytesIO(cols=30, size=self.current_step*GIGA)).compute()
    @bench(name="Naive", corrected_by="Nop")
    def naive_read_csv(self):
        res = {}
        reader = RandomBytesIO(cols=30, size=self.current_step*GIGA)
        first = next(reader)
        for i, cell in enumerate(first.split(',')):
            res[i] = [float(cell)]
        for row in reader:
            for i, cell in enumerate(row.split(',')):
                res[i].append(float(cell))
        return res
    def runTest(self):
        self.runBench()
        self.save(db_name='bench_refs.db', name='load_csv')

if __name__ == '__main__':
    unittest.main()
