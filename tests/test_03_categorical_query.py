from __future__ import annotations

from . import ProgressiveTest
from progressivis.core.api import Sink
from progressivis.core import aio
from progressivis.io import SimpleCSVLoader
from progressivis.table.categorical_query import CategoricalQuery
from progressivis.table.constant import ConstDict
from progressivis.utils.psdict import PDict
import pandas as pd
import numpy as np
from io import StringIO
from typing import Sequence, Tuple


def generate_random_csv(
    rows: int = 300_000, seed: int = 42, choice: Sequence[str] = ("A", "B", "C", "D"),
) -> Tuple[pd.DataFrame, str]:
    np.random.seed(seed)
    df = pd.DataFrame(
        {
            "I": np.random.randint(0, 10_000, size=rows, dtype=int),
            "J": np.random.randint(0, 15_000, size=rows, dtype=int),
            "category": np.random.choice(choice, rows),
        }
    )
    sio = StringIO()
    df.to_csv(sio, index=False)
    sio.seek(0)
    return df, sio.getvalue()


df_, csv_ = generate_random_csv()
a_c = df_.query("category=='A' or category=='C'")


class TestProgressiveCatQuery(ProgressiveTest):
    def test_cat_query(self) -> None:
        s = self.scheduler()
        sio = StringIO(csv_)
        sio.seek(0)
        csv = SimpleCSVLoader(sio, scheduler=s,)
        query = CategoricalQuery(column="category", scheduler=s)
        query.create_dependent_modules(
            input_module=csv
        )
        ct = ConstDict(PDict({"only": ["A", "C"]}), scheduler=s)
        query.input.choice = ct.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = query.output.result
        aio.run(s.start())
        assert query.result is not None
        df = query.result.to_df()
        self.assertTrue(df.equals(a_c.reset_index(drop=True)))
