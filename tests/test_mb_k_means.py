import unittest

from progressivis import Scheduler, Print, log_level
from progressivis.cluster import MBKMeans
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset


from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.extmath import squared_norm

import numpy as np
import pandas as pd

times = 0

def stop_if_done(s, n):
    global times
    if s.run_queue_length()==2:
        if times==2:
            s.stop()
        times += 1


class TestMBKmeans(unittest.TestCase):
    def test_mb_k_means(self):
        #log_level()
        s=Scheduler()
        csv = CSVLoader(get_dataset('cluster:s3'),sep='    ',skipinitialspace=True,header=None,index_col=False,scheduler=s)
        km = MBKMeans(n_clusters=3, random_state=42, scheduler=s)
        km.input.df = csv.output.df
        pr = Print(scheduler=s)
        pr.input.df = km.output.centroids
        s.start(idle_proc=stop_if_done)
        self.assertEquals(len(csv.df()), len(km.df()))
        import pdb
        pdb.set_trace()
        mbk = MiniBatchKMeans(n_clusters=3, random_state=42, verbose=True)
        X = csv.df()[km.columns]
        mbk.partial_fit(X)
        print mbk.cluster_centers_
        print km.mbk.cluster_centers_
        self.assertTrue(np.allclose(mbk.cluster_centers_, km.mbk.cluster_centers_))


if __name__ == '__main__':
    unittest.main()

