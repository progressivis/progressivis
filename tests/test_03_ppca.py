from . import ProgressiveTest, skipIf
from progressivis import Scheduler, Every
from progressivis.core import aio
from progressivis.io import CSVLoader
from progressivis.stats.ppca import PPCA
from progressivis.datasets import get_dataset
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from progressivis.utils.psdict import PsDict
from progressivis.core.slot import SlotDescriptor
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.utils.random import sample_without_replacement  # type: ignore

import pandas as pd
import os


def _print(x):
    pass


TRAIN_SAMPLE_SIZE = 10000
PREDICT_SAMPLE_SIZE = 1000
SAMPLE_SIZE = TRAIN_SAMPLE_SIZE + PREDICT_SAMPLE_SIZE
RANDOM_STATE = 42
NNEIGHBOURS = 7
N_COMPONENTS = 154
TRACE = None  # 'verbose'
LABELS = INDICES = KNN = None


def _array(tbl):
    return tbl["array"].values


class MyResetter(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, threshold, **kwds):
        super().__init__(**kwds)
        self._threshold = threshold
        self.result = PsDict({"reset": True})

    def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot("table")
        input_slot.clear_buffers()
        data = input_slot.data()
        if data and len(data) >= self._threshold:
            self.result["reset"] = False
        return self._return_run_step(self.next_state(input_slot), steps_run=step_size)


@skipIf(os.getenv("CI"), "skipped because too expensive for the CI")
class TestPPCA(ProgressiveTest):
    def _common(
        self, rtol, threshold=None, resetter=None, resetter_func=None, scheduler=None
    ):
        global KNN, LABELS, INDICES
        if scheduler is None:
            s = Scheduler()
        else:
            s = scheduler
        dataset = get_dataset("mnist_784")
        data = CSVLoader(
            dataset,
            index_col=False,
            as_array="array",
            usecols=lambda x: x != "class",
            scheduler=s,
        )
        ppca = PPCA(scheduler=s)
        ppca.input[0] = data.output.result
        ppca.params.n_components = N_COMPONENTS
        if resetter:
            assert callable(resetter_func)
            resetter.input[0] = ppca.output.result
        ppca.create_dependent_modules(
            rtol=rtol,
            trace=TRACE,
            threshold=threshold,
            resetter=resetter,
            resetter_func=resetter_func,
        )

        prn = Every(scheduler=s, proc=_print)
        prn.input[0] = ppca.reduced.output.result
        aio.run(s.start())
        pca_ = ppca._transformer["inc_pca"]
        recovered = pca_.inverse_transform(_array(ppca.reduced.result))
        if KNN is None:
            print("Init KNN")
            KNN = KNeighborsClassifier(NNEIGHBOURS)
            arr = _array(data.result)
            LABELS = pd.read_csv(dataset, usecols=["class"]).values.reshape((-1,))
            indices_t = sample_without_replacement(
                n_population=len(data.result),
                n_samples=TRAIN_SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            )
            KNN.fit(arr[indices_t], LABELS[indices_t])
        indices_p = sample_without_replacement(
            n_population=len(data.result),
            n_samples=PREDICT_SAMPLE_SIZE,
            random_state=RANDOM_STATE * 2 + 1,
        )
        return KNN.score(recovered[indices_p], LABELS[indices_p])

    def test_always_reset(self):
        """
        test_always_reset()
        """
        score = self._common(0.1)
        print("always reset=>score", score)
        self.assertGreater(score, 0.93)  # 0.94?

    def test_never_reset(self):
        """
        test_never_reset()
        """
        score = self._common(100.0)
        print("never reset=>score", score)
        self.assertGreater(score, 0.77)

    def test_reset_threshold_30k(self):
        """
        test_reset_threshold_30k ()
        """
        score = self._common(0.1, threshold=30000)
        print("reset when threshold 30K=>score", score)
        self.assertGreater(score, 0.77)

    def test_reset_threshold_40k(self):
        """
        test_reset_threshold_40k()
        """
        score = self._common(0.1, threshold=40000)
        print("reset when threshold 40K=>score", score)
        self.assertGreater(score, 0.77)

    def test_reset_threshold_50k(self):
        """
        test_reset_threshold_50k()
        """
        score = self._common(0.1, threshold=50000)
        print("reset when threshold 50K=>score", score)
        self.assertGreater(score, 0.77)

    def test_reset_threshold_60k(self):
        """
        test_reset_threshold_60k()
        """
        score = self._common(0.1, threshold=60000)
        print("reset when threshold 60K=>score", score)
        self.assertGreater(score, 0.77)

    def test_resetter(self):
        """
        test_resetter()
        """
        s = Scheduler()
        resetter = MyResetter(threshold=30000, scheduler=s)

        def _func(slot):
            return slot.data().get("reset")

        score = self._common(0.1, resetter=resetter, resetter_func=_func, scheduler=s)
        print("resetter 30K=>score", score)
        self.assertGreater(score, 0.77)


if __name__ == "__main__":
    TestPPCA.main()
