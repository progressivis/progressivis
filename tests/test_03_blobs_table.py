from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Every
from progressivis.stats.blobs_table import BlobsTable, MVBlobsTable
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
        self.assertEqual(module.result.columns[0],'a')
        self.assertEqual(module.result.columns[1],'b')
        self.assertEqual(len(module.result.columns), 2)
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = module.output.result
        aio.run(s.start())
        #s.join()
        self.assertEqual(len(module.result), 10000)

    def test_blobs_table2(self):
        s = self.scheduler()
        sz = 100000
        centers = [(0.1, 0.3), (0.7, 0.5), (-0.4, -0.3)]
        blob1=BlobsTable(['a', 'b'], centers=centers,  cluster_std=0.2, rows=sz, scheduler=s)
        blob1.default_step_size = 1500
        blob2=BlobsTable(['a', 'b'], centers=centers,  cluster_std=0.2, rows=sz, scheduler=s)
        blob2.default_step_size = 200
        add = Add(scheduler=s)
        add.input.first = blob1.output.result
        add.input.second = blob2.output.result
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = add.output.result
        aio.run(s.start())
        #s.join()
        self.assertEqual(len(blob1.result), sz)
        self.assertEqual(len(blob2.result), sz)
        arr1 = blob1.result.to_array()
        arr2 = blob2.result.to_array()
        self.assertTrue(np.allclose(arr1, arr2))

means = [0.1, 0.3], [0.7, 0.5], [-0.4, -0.3]
covs = [[0.01, 0], [0, 0.09]], [[0.04, 0], [0, 0.01]], [[0.09, 0.04], [0.04, 0.02]]

class TestMVBlobsTable(ProgressiveTest):
    def test_mv_blobs_table(self):
        s = self.scheduler()
        module=MVBlobsTable(['a', 'b'], means=means, covs=covs, rows=10000, scheduler=s)
        self.assertEqual(module.result.columns[0],'a')
        self.assertEqual(module.result.columns[1],'b')
        self.assertEqual(len(module.result.columns), 2)
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = module.output.result
        aio.run(s.start())
        #s.join()
        self.assertEqual(len(module.result), 10000)

    def test_mv_blobs_table2(self):
        s = self.scheduler()
        sz = 100000
        blob1=MVBlobsTable(['a', 'b'], means=means, covs=covs, rows=sz, scheduler=s)
        blob1.default_step_size = 1500
        blob2=MVBlobsTable(['a', 'b'], means=means, covs=covs, rows=sz, scheduler=s)
        blob2.default_step_size = 200
        add = Add(scheduler=s)
        add.input.first = blob1.output.result
        add.input.second = blob2.output.result
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = add.output.result
        aio.run(s.start())
        #s.join()
        self.assertEqual(len(blob1.result), sz)
        self.assertEqual(len(blob2.result), sz)
        arr1 = blob1.result.to_array()
        arr2 = blob2.result.to_array()
        self.assertTrue(np.allclose(arr1, arr2))
