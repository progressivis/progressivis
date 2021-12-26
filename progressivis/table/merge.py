"Merge module."
from __future__ import annotations

from .nary import NAry, ReturnRunStep
from .table import BaseTable, Table
from .dshape import dshape_join
from progressivis.utils.inspect import filter_kwds

from typing import Dict, Any, cast, List


def merge(
    left: BaseTable,
    right: BaseTable,
    name: str = None,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=False,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    merge_ctx: Dict[str, Any] = None,
) -> Table:
    # pylint: disable=too-many-arguments, invalid-name, unused-argument, too-many-locals
    "Merge function"
    lsuffix, rsuffix = suffixes
    if not all((left_index, right_index)):
        raise ValueError(
            "currently, only right_index=True and "
            "left_index=True are allowed in Table.merge()"
        )
    dshape, rename = dshape_join(left.dshape, right.dshape, lsuffix, rsuffix)
    merge_table = Table(name=name, dshape=dshape)
    if how == "inner":
        merge_ids = left.index & right.index
        new_ids = left.index & merge_ids
        merge_table.resize(len(new_ids), index=new_ids)
        left_cols = [rename["left"].get(c, c) for c in left.columns]
        right_cols = [rename["right"].get(c, c) for c in right.columns]
        merge_table.loc[merge_ids, left_cols] = left.loc[merge_ids, left.columns]
        merge_table.loc[merge_ids, right_cols] = right.loc[merge_ids, right.columns]
    else:
        raise ValueError("how={} not implemented in Table.merge()".format(how))
    if isinstance(merge_ctx, dict):
        merge_ctx["dshape"] = dshape
        merge_ctx["left_cols"] = left_cols
        merge_ctx["right_cols"] = right_cols
    return merge_table


def merge_cont(left: BaseTable,
               right: BaseTable,
               merge_ctx: Dict[str, Any]) -> Table:
    "merge continuation function"
    merge_table = Table(name=None, dshape=merge_ctx["dshape"])
    merge_ids = left.index & right.index
    new_ids = left.index & merge_ids
    merge_table.resize(len(new_ids), index=new_ids)
    merge_table.loc[merge_ids, merge_ctx["left_cols"]] = left.loc[
        merge_ids, left.columns
    ]
    merge_table.loc[merge_ids, merge_ctx["right_cols"]] = right.loc[
        merge_ids, right.columns
    ]
    return merge_table


class Merge(NAry):
    "Merge module"

    def __init__(self, **kwds):
        """Merge(how='inner', on=None, left_on=None, right_on=None,
                 left_index=False, right_index=False,
                sort=False,suffixes=('_x', '_y'), copy=True,
                indicator=False)
        """
        super(Merge, self).__init__(**kwds)
        self.merge_kwds = filter_kwds(kwds, merge)
        self._context = {}

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        frames: List[BaseTable] = []
        for name in self.get_input_slot_multiple():
            slot = self.get_input_slot(name)
            df = cast(BaseTable, slot.data())
            slot.clear_buffers()
            frames.append(df)
        df = frames[0]
        for other in frames[1:]:
            if not self._context:
                df = merge(df, other, merge_ctx=self._context, **self.merge_kwds)
            else:
                df = merge_cont(df, other, merge_ctx=self._context)
        length = len(df)
        self.result = df
        return self._return_run_step(self.state_blocked, steps_run=length)
