from . import ProgressiveTest

from progressivis import Print, Every #, log_level
from progressivis.cluster import MBKMeans
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset
from progressivis.core import aio


#from sklearn.cluster import MiniBatchKMeans
#from sklearn.utils.extmath import squared_norm

#import numpy as np
#import pandas as pd

# times = 0

# def stop_if_done(s, n):
#     global times
#     if s.run_queue_length()==3:
#         if times==2:
#             s.stop()
#         times += 1


class TestMBKmeans(ProgressiveTest):
    def test_mb_k_means(self):
        #log_level()
        s = self.scheduler()
        n_clusters = 3
        with s:
            csv = CSVLoader(get_dataset('cluster:s3'),
                            sep=' ',
                            skipinitialspace=True,
                            header=None,
                            index_col=False,
                            scheduler=s)
            km = MBKMeans(n_clusters=n_clusters, random_state=42,
                          is_input=False, is_greedy=False, scheduler=s)
            #km.input.table = csv.output.result
            km.create_dependent_modules(csv)
            pr = Print(proc=self.terse, scheduler=s)
            pr.input[0] = km.output.result
            e = Every(proc=self.terse, scheduler=s)
            e.input[0] = km.output.labels
        aio.run(s.start())
        # s.join()
        self.assertEqual(len(csv.result), len(km.labels()))
        # mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, verbose=True)
        # X = csv.df()[km.columns]
        # mbk.partial_fit(X)
        # print mbk.cluster_centers_
        # print km.mbk.cluster_centers_
        # self.assertTrue(np.allclose(mbk.cluster_centers_, km.mbk.cluster_centers_))


if __name__ == '__main__':
    ProgressiveTest.main()

