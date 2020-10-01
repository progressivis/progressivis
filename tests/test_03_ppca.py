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

SAMPLE_SIZE = 10000
RANDOM_STATE = 42
NNEIGHBOURS = 7
N_COMPONENTS = 154
TRACE = None # 'verbose'

#@skip
class TestPPCA(ProgressiveTest):
    def _common(self, rtol):
        s = Scheduler()        
        dataset = get_dataset('mnist_784')
        data = CSVLoader(dataset, index_col=False,
                     usecols=lambda x: x!='class', scheduler=s)
        ppca = PPCA(scheduler=s)
        ppca.input.table = data.output.table
        ppca.params.n_components = N_COMPONENTS
        ppca.create_dependent_modules(rtol=rtol, trace=TRACE)
        prn = Every(scheduler=s, proc=_print)
        prn.input.df = ppca.reduced.output.table    
        aio.run(s.start())
        pca_ = ppca._transformer['inc_pca']
        recovered = pca_.inverse_transform(ppca.reduced._table.to_array())
        knn = KNeighborsClassifier(NNEIGHBOURS)
        arr = data._table.to_array()
        labels = pd.read_csv(dataset, usecols=['class']).values.reshape((-1,))
        indices = sample_without_replacement(n_population=arr.shape[0],
                                             n_samples=SAMPLE_SIZE,
                                             random_state=RANDOM_STATE)
        knn.fit(arr[indices], labels[indices])
        return knn.score(recovered[indices], labels[indices])

    def test_always_reset(self):
        """
        test_always_reset()
        """
        score = self._common(0.1)
        print("always reset=>score", score)
        self.assertGreater(score, 0.95)

    def test_never_reset(self):
        """
        test_never_reset()
        """
        score = self._common(100.0)
        print("never reset=>score", score)
        self.assertGreater(score, 0.8)
        


if __name__ == '__main__':
    unittest.main()
