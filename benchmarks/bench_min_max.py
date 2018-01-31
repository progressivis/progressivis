import pandas as pd
import csv
from progressivis import Scheduler
from progressivis.io import CSVLoader
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.datasets import get_dataset
from benchmarkit import BenchEnv, BenchmarkIt, InputProc
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

import dask.dataframe as dd

def cleanup_hdf5():
    for d in glob.glob('/var/tmp/progressivis_*'):
        subprocess.call(['/bin/rm','-rf', d])
    for d in glob.glob('/tmp/progressivis_*'):
        subprocess.call(['/bin/rm','-rf', d])



def make_df(n, L):
    s=Scheduler()
    random = RandomTable(10, rows=n*L, scheduler=s)
    s.start()
    #return random
    return pd.DataFrame(random.output.table.output_module.table().to_dict())


if __name__=='__main__':
    if len(sys.argv) != 5:
        print("Usage {} <dbname> N L R".format(sys.argv[0]))
        sys.exit()
    db_name = sys.argv[1]
    N = int(sys.argv[2])
    L = int(sys.argv[3])
    R = int(sys.argv[4])        
    main_loop = range(1, N+1)
    df_list = []
    for i in main_loop:
        df_list.append( make_df(i, L))
        cleanup_hdf5()
    benv = BenchEnv(db_name=db_name)
    with BenchmarkIt(env=benv, name="MinMax", desc="Min, Max measurements",
                     repeat=R,
                     input_proc=InputProc(repr_type=int, label='X{} rows'.format(L)),
                     loop=main_loop, time_bm=True, memory_bm=True,
                     after_loop_func=cleanup_hdf5) as bench:
        
        @bench(case="P10sNPCorrector")
        def p10s_np_random(n):
            StorageEngine.default = "numpy"
            s=Scheduler()
            random = RandomTable(10, rows=n*L, scheduler=s)
            s.start()

        @bench(case="P10sNPMinMax", corrected_by="P10sNPCorrector")
        def p10s_np_random_min_max(n):
            StorageEngine.default = "numpy"
            s=Scheduler()
            random = RandomTable(10, rows=n*L, scheduler=s)
            min_=Min(mid='min_'+str(hash(random)), scheduler=s)
            min_.input.table = random.output.table
            max_=Max(id='max_'+str(hash(random)), scheduler=s)
            max_.input.table = random.output.table
            s.start()

        @bench(case="P10sH5Corrector")
        def p10s_random(n):
            StorageEngine.default = "hdf5"            
            s=Scheduler()
            random = RandomTable(10, rows=n*L, scheduler=s)
            s.start()

        @bench(case="P10sH5MinMax", corrected_by="P10sH5Corrector")
        def p10s_random_min_max(n):
            StorageEngine.default = "hdf5"            
            s=Scheduler()
            random = RandomTable(10, rows=n*L, scheduler=s)
            min_=Min(mid='min_'+str(hash(random)), scheduler=s)
            min_.input.table = random.output.table
            max_=Max(id='max_'+str(hash(random)), scheduler=s)
            max_.input.table = random.output.table
            s.start()

        #@bench(case="P10sZarrCorrector")
        def p10s_zarr_random(n):
            StorageEngine.default = "zarr"
            s=Scheduler()
            random = RandomTable(10, rows=n*L, scheduler=s)
            s.start()

        #@bench(case="P10sZarrMinMax", corrected_by="P10sZarrCorrector")
        def p10s_zarr_random_min_max(n):
            StorageEngine.default = "zarr"
            s=Scheduler()
            random = RandomTable(10, rows=n*L, scheduler=s)
            min_=Min(mid='min_'+str(hash(random)), scheduler=s)
            min_.input.table = random.output.table
            max_=Max(id='max_'+str(hash(random)), scheduler=s)
            max_.input.table = random.output.table
            s.start()

        #@bench(case="P10sBcolzCorrector")
        def p10s_bcolz_random(n):
            StorageEngine.default = "bcolz"
            s=Scheduler()
            random = RandomTable(10, rows=n*L, scheduler=s)
            s.start()

        #@bench(case="P10sBcolzMinMax", corrected_by="P10sBcolzCorrector")
        def p10s_bcolz_random_min_max(n):
            StorageEngine.default = "bcolz"
            s=Scheduler()
            random = RandomTable(10, rows=n*L, scheduler=s)
            min_=Min(mid='min_'+str(hash(random)), scheduler=s)
            min_.input.table = random.output.table
            max_=Max(id='max_'+str(hash(random)), scheduler=s)
            max_.input.table = random.output.table
            s.start()
            
        @bench(case="Pandas")
        def pandas_min_max(n):
            df = df_list[n-1]
            df.min()
            df.max()


