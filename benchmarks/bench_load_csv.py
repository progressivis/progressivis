import pandas as pd
import csv
from progressivis import Scheduler
from progressivis.io import CSVLoader
from benchmarkit import BenchEnv, BenchmarkIt
import sys
import subprocess
import glob

import dask.dataframe as dd

def cleanup_hdf5():
    for d in glob.glob('/tmp/progressivis_*'):
        subprocess.call(['/bin/rm','-rf', d])



if __name__=='__main__':
    if len(sys.argv) < 3:
        print("Usage {} <dbname> <csvfile1> [<csvfile2>...<csvfileN>]".format(sys.argv[0]))
        sys.exit()
    db_name = sys.argv[1]
    benv = BenchEnv(db_name=db_name)


    with BenchmarkIt(env=benv, name="Load CSV", desc="Load CSV measurements",
                     loop=sys.argv[2:], after_loop_func=cleanup_hdf5) as bench:
        @bench(case="Nop")
        def none_read_csv(f):
            with open(f, 'rb') as csvfile:
                for _ in csvfile:
                    pass
        @bench("Progressivis", corrected_by="Nop")
        def p10s_read_csv(f):
            s=Scheduler()
            module=CSVLoader(f, index_col=False, header=None, scheduler=s)
            module.start()
        @bench("Pandas", corrected_by="Nop")
        def pandas_read_csv(f):
            pd.read_csv(f)
        @bench("Dask", corrected_by="Nop")
        def dask_read_csv(f):
            dd.read_csv(f).compute()
        @bench("Naive", corrected_by="Nop")
        def naive_read_csv(f):
            res = {}
            with open(f, 'rt') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                first = next(reader)
                for i, cell in enumerate(first):
                    res[i] = [float(cell)]
                for row in reader:
                    for i, cell in enumerate(row):
                        res[i].append(float(cell))
            return res


