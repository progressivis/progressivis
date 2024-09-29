from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Print
from progressivis.stats.quantiles import Quantiles
from progressivis.stats import RandomPTable

import numpy as np
# from datasketches import kll_floats_sketch
from typing import Any, Sequence, Union

K = 300
BINS = 128
QUANTILES = [0.3, 0.5, 0.7]
NAMED_QUANTILES = ["first", "second", "third"]
SPLITS_SEQ = [0.3, 0.5, 0.7]
SPLITS_DICT = dict(lower=0.1, upper=0.9, n_splits=10)

ArrayLike = Union[np.ndarray[Any, Any], Sequence[Any]]

RNG = np.random.default_rng()


def uniform(n: int) -> np.ndarray[Any, Any]:
    return RNG.uniform(low=-100, high=100, size=n)


class TestQuantiles(ProgressiveTest):
    def test_quantiles(self) -> None:
        np.random.seed(42)
        s = self.scheduler()
        random = RandomPTable(
            3,
            rows=10_000,
            random=uniform,
            scheduler=s
        )
        quantiles = Quantiles(k=K, scheduler=s)
        quantiles.input.table = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = quantiles.output.result[0.5]  # print the median
        aio.run(s.start())
        assert random.result is not None
        assert quantiles.result is not None
        # val = random.result["_1"].value
        # sk = kll_floats_sketch(K)
        # sk.update(val)
        self.assertEqual(quantiles.result["_1"], len(random.result))
        self.assertEqual(quantiles.result["_2"], len(random.result))
        self.assertEqual(quantiles.result["_3"], len(random.result))
        median = quantiles.get_data("result", 0.5)
        self.assertLess(np.abs(median["_1"]), 4)  # empirical error for 10,000 items
        self.assertLess(np.abs(median["_2"]), 4)
        self.assertLess(np.abs(median["_3"]), 4)


if __name__ == "__main__":
    ProgressiveTest.main()
