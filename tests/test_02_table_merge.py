from collections import OrderedDict

import pandas as pd

from progressivis import Scheduler
from progressivis.table.table import Table
from progressivis.table.merge import merge

from . import ProgressiveTest

df_left1 = pd.DataFrame(
    OrderedDict(
        [
            ("A", ["A0", "A1", "A2", "A3"]),
            ("B", ["B0", "B1", "B2", "B3"]),
            ("C", ["C0", "C1", "C2", "C3"]),
        ]
    ),
    index=[0, 1, 2, 3],
)
df_right1 = pd.DataFrame(
    {
        "X": ["X0", "X1", "X2", "X3"],
        "Y": ["Y0", "Y2", "Y3", "Y4"],
        "Z": ["Z0", "Z2", "Z3", "Z4"],
    },
    index=[2, 3, 4, 5],
)

df_right2 = pd.DataFrame(
    OrderedDict(
        [
            ("X", ["X0", "X1", "X2", "X3"]),
            ("Y", ["Y0", "Y2", "Y3", "Y4"]),
            ("B", ["Br0", "Br2", "Br3", "Br4"]),
            ("Z", ["Z0", "Z2", "Z3", "Z4"]),
        ]
    ),
    index=[2, 3, 4, 5],
)


class TestMergeTable(ProgressiveTest):
    def setUp(self) -> None:
        super(TestMergeTable, self).setUp()
        self.scheduler_ = Scheduler.default

    def test_merge1(self) -> None:
        table_left = Table(name="table_left", data=df_left1, create=True)
        print(repr(table_left))
        table_right = Table(
            name="table_right",
            data=df_right1,
            create=True,
            indices=df_right1.index.values,
        )
        print(repr(table_right))
        # table_right2 = Table(name='table_right2', data=df_right2, create=True)
        table_merge = merge(
            table_left,
            table_right,
            name="table_merge",
            left_index=True,
            right_index=True,
        )
        print(repr(table_merge))


if __name__ == "__main__":
    ProgressiveTest.main()
