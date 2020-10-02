from . import ProgressiveTest, skip, skipIf
from progressivis import Print, Scheduler, Every
from progressivis.core import aio
from progressivis.io import CSVLoader
from progressivis.stats.ppca import PPCA
from progressivis.datasets import get_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.random import sample_without_replacement
import numpy as np
import pandas as pd

def _print(x):
    pass

TRAIN_SAMPLE_SIZE = 10000
PREDICT_SAMPLE_SIZE = 1000
SAMPLE_SIZE = TRAIN_SAMPLE_SIZE + PREDICT_SAMPLE_SIZE
RANDOM_STATE = 42
NNEIGHBOURS = 7
N_COMPONENTS = 154
TRACE = None #'verbose'
LABELS = INDICES = KNN = None

#@skip
class TestPPCA(ProgressiveTest):
    def _common(self, rtol, threshold=None):
        global KNN, LABELS, INDICES
        s = Scheduler()        
        dataset = get_dataset('mnist_784')
        data = CSVLoader(dataset, index_col=False,
                     usecols=lambda x: x!='class', scheduler=s)
        ppca = PPCA(scheduler=s)
        ppca.input.table = data.output.table
        ppca.params.n_components = N_COMPONENTS
        ppca.create_dependent_modules(rtol=rtol, trace=TRACE, threshold=threshold)
        prn = Every(scheduler=s, proc=_print)
        prn.input.df = ppca.reduced.output.table    
        aio.run(s.start())
        #import pdb;pdb.set_trace()
        pca_ = ppca._transformer['inc_pca']
        recovered = pca_.inverse_transform(ppca.reduced._table.to_array())
        if KNN is None:
            print("Init KNN")
            KNN = KNeighborsClassifier(NNEIGHBOURS)
            arr = data._table.to_array()
            LABELS = pd.read_csv(dataset, usecols=['class']).values.reshape((-1,))
            INDICES = sample_without_replacement(n_population=len(data._table),
                                                 n_samples=SAMPLE_SIZE,
                                                 random_state=RANDOM_STATE)
            indices_t = INDICES[:TRAIN_SAMPLE_SIZE]
            KNN.fit(arr[indices_t], LABELS[indices_t])
        indices_p = INDICES[TRAIN_SAMPLE_SIZE:]
        return KNN.score(recovered[indices_p], LABELS[indices_p])

    def test_always_reset(self):
        """
        test_always_reset()
        """
        score = self._common(0.1)
        print("always reset=>score", score)
        self.assertGreater(score, 0.94)

    def test_never_reset(self):
        """
        test_never_reset()
        """
        score = self._common(100.0)
        print("never reset=>score", score)
        self.assertGreater(score, 0.79)
        
    def test_reset_threshold_30k(self):
        """
        test_reset_threshold_30k ()
        """
        score = self._common(0.1, threshold=30000)
        print("reset when threshold 30K=>score", score)
        self.assertGreater(score, 0.79)

    def test_reset_threshold_40k(self):
        """
        test_reset_threshold_40k()
        """
        score = self._common(0.1, threshold=40000)
        print("reset when threshold 40K=>score", score)
        self.assertGreater(score, 0.79)
        
    def test_reset_threshold_50k(self):
        """
        test_reset_threshold_50k()
        """
        score = self._common(0.1, threshold=50000)
        print("reset when threshold 50K=>score", score)
        self.assertGreater(score, 0.79)
        
    def test_reset_threshold_60k(self):
        """
        test_reset_threshold_60k()
        """
        score = self._common(0.1, threshold=60000)
        print("reset when threshold 60K=>score", score)
        self.assertGreater(score, 0.79)
        


if __name__ == '__main__':
    unittest.main()
