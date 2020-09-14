from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Every
from progressivis.stats.blobs_table import BlobsTable
from progressivis.stats.random_table import RandomTable
from progressivis.linalg import Add
import numpy as np
from progressivis.io import CSVLoader
def print_len(x):
    if x is not None:
        print(len(x))

centers = [(0.1, 0.3), (0.7, 0.5), (-0.4, -0.3)]
class TestBlobsTable(ProgressiveTest):
    def test_blobs_table(self):
        s = self.scheduler()
        module=BlobsTable(['a', 'b'], centers=centers, rows=10000, scheduler=s)
        self.assertEqual(module.table().columns[0],'a')
        self.assertEqual(module.table().columns[1],'b')
        self.assertEqual(len(module.table().columns), 2) 
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input.df = module.output.table
        aio.run(s.start())
        #s.join()
        self.assertEqual(len(module.table()), 10000)

    def test_blobs_table2(self):
        s = self.scheduler()
        sz = 100000
        centers = [(0.1, 0.3), (0.7, 0.5), (-0.4, -0.3)]
        blob1=BlobsTable(['a', 'b'], centers=centers,  cluster_std=0.2, rows=sz, scheduler=s)
        #blob1.default_step_size = 1500
        blob2=BlobsTable(['a', 'b'], centers=centers,  cluster_std=0.2, rows=sz, scheduler=s)
        #blob2.default_step_size = 200
        add = Add(scheduler=s)
        add.input.first = blob1.output.table
        add.input.second = blob2.output.table        
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input.df = add.output.table
        aio.run(s.start())
        #s.join()
        self.assertEqual(len(blob1.table()), sz)
        self.assertEqual(len(blob2.table()), sz)
        arr1 = blob1.table().to_array()
        arr2 = blob2.table().to_array()
        self.assertTrue(np.allclose(arr1, arr2))
