"Merge module."
from __future__ import annotations

from progressivis.core.module import Module, ReturnRunStep
from .table_base import BasePTable
from .table import PTable
from .dshape import dshape_join
from progressivis.utils.inspect import filter_kwds

from typing import Dict, Any, cast, List, Tuple, Optional


def merge(
    left: BasePTable,
    right: BasePTable,
    name: Optional[str] = None,
    how: str = "inner",
    on: Any = None,
    left_on: Any = None,
    right_on: Any = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    copy: bool = True,
    indicator: bool = False,
    merge_ctx: Optional[Dict[str, Any]] = None,
) -> PTable:
    # pylint: disable=too-many-arguments, invalid-name, unused-argument, too-many-locals
    "Merge function"
    lsuffix, rsuffix = suffixes
    if not all((left_index, right_index)):
        raise ValueError(
            "currently, only right_index=True and "
            "left_index=True are allowed in PTable.merge()"
        )
    dshape, rename = dshape_join(left.dshape, right.dshape, lsuffix, rsuffix)
    merge_table = PTable(name=name, dshape=dshape)
    if how == "inner":
        merge_ids = left.index & right.index
        new_ids = left.index & merge_ids
        merge_table.resize(len(new_ids), index=new_ids)
        left_cols = [rename["left"].get(c, c) for c in left.columns]
        right_cols = [rename["right"].get(c, c) for c in right.columns]
        merge_table.loc[merge_ids, left_cols] = left.loc[merge_ids, left.columns]
        merge_table.loc[merge_ids, right_cols] = right.loc[merge_ids, right.columns]
    else:
        raise ValueError("how={} not implemented in PTable.merge()".format(how))
    if isinstance(merge_ctx, dict):
        merge_ctx["dshape"] = dshape
        merge_ctx["left_cols"] = left_cols
        merge_ctx["right_cols"] = right_cols
    return merge_table


def merge_cont(left: BasePTable, right: BasePTable, merge_ctx: Dict[str, Any]) -> PTable:
    "merge continuation function"
    merge_table = PTable(name=None, dshape=merge_ctx["dshape"])
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


class Merge(Module):
    "Merge module"

    def __init__(self, **kwds: Any) -> None:
        """Merge(how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False,
        sort=False,suffixes=('_x', '_y'), copy=True,
        indicator=False)
        """
        super().__init__(**kwds)
        self.merge_kwds = filter_kwds(kwds, merge)
        self._context: Dict[str, Any] = {}

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        frames: List[BasePTable] = []
        for name in self.get_input_slot_multiple("table"):
            slot = self.get_input_slot(name)
            df = cast(BasePTable, slot.data())
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
