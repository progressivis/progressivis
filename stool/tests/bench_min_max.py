import pandas as pd
import csv
from progressivis import Scheduler
from progressivis.io import CSVLoader
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.datasets import get_dataset
from progressivis.core.storage import StorageEngine
#from progressivis.core.storage.zarr import ZARRStorageEngine
import sys
import os
import os.path
import subprocess
import glob
from collections import OrderedDict
from progressivis import Print, Scheduler
from progressivis.stats import Min, Max, RandomTable
from stool import BenchmarkCase, bench


import dask.dataframe as dd

import unittest


L = 1000
        
class BenchMinMax(BenchmarkCase):
    def __init__(self, testMethod='runBench'):
        super(BenchMinMax, self).__init__(testMethod)
        self.nb_step = 8
        self.random_table = None
        self.db_name = 'bench_refs.db'
        self.bench_name = 'min_max'
        self.with_code_prof = True
    def tearDown(self):
        for d in glob.glob('/var/tmp/progressivis_*'):
            subprocess.call(['/bin/rm','-rf', d])
        for d in glob.glob('/tmp/progressivis_*'):
            subprocess.call(['/bin/rm','-rf', d])
    def setUp(self):
        self.set_step_header("Processed rows (x {})".format(L))            
    def setUpStep(self, step):
        self.set_step_info("{} rows".format(step*L))
        s=Scheduler()
        random = RandomTable(10, rows=step*L, scheduler=s)
        s.start()
        #return random
        self.random_table = pd.DataFrame(
            random.output.table.output_module.table().to_dict())
        
    def tearDownStep(self, step):
        self.random_table = None
        
    @bench(name="P10sNPCorrector")
    def p10s_np_random(self):
        n = self.current_step        
        StorageEngine.default = "numpy"
        s=Scheduler()
        random = RandomTable(10, rows=n*L, scheduler=s)
        s.start()

    @bench(name="P10sNPMinMax", corrected_by="P10sNPCorrector")
    def p10s_np_random_min_max(self):
        n = self.current_step
        StorageEngine.default = "numpy"
        s=Scheduler()
        random = RandomTable(10, rows=n*L, scheduler=s)
        min_=Min(mid='min_'+str(hash(random)), scheduler=s)
        min_.input.table = random.output.table
        max_=Max(id='max_'+str(hash(random)), scheduler=s)
        max_.input.table = random.output.table
        s.start()

    @bench(name="P10sH5Corrector")
    def p10s_random(self):
        n = self.current_step        
        StorageEngine.default = "hdf5"            
        s=Scheduler()
        random = RandomTable(10, rows=n*L, scheduler=s)
        s.start()

    @bench(name="P10sH5MinMax", corrected_by="P10sH5Corrector")
    def p10s_random_min_max(self):
        n = self.current_step
        StorageEngine.default = "hdf5"            
        s=Scheduler()
        random = RandomTable(10, rows=n*L, scheduler=s)
        min_=Min(mid='min_'+str(hash(random)), scheduler=s)
        min_.input.table = random.output.table
        max_=Max(id='max_'+str(hash(random)), scheduler=s)
        max_.input.table = random.output.table
        s.start()

    @bench(name="Pandas")
    def pandas_min_max(self):
        df = self.random_table
        df.min()
        df.max()
        
    def runTest(self):
        self.runBench()
        other = None
        if os.path.exists(self.db_name):
            other = BenchmarkCase.load(self.db_name, self.bench_name)
        if other is None: # other could be None id DB exists but not the bench
            self.save(db_name=self.db_name, name=self.bench_name)
            print("New benchmark {} created on {} DB!".format(self.bench_name, self.db_name))
            return
        # mem_usage with pandas, step=1, repeat=0 on self
        self_mem_pandas_s1_r0 = self['Pandas'].query('step == 1')['mem_usage'].values[0]
        # same stuff on other
        other_mem_pandas_s1_r0 = other['Pandas'].query('step == 1')['mem_usage'].values[0]
        self.assertAlmostEqual(self_mem_pandas_s1_r0, other_mem_pandas_s1_r0, delta=self_mem_pandas_s1_r0*0.1)

if __name__ == '__main__':
    unittest.main()
