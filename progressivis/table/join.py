"Join Module."
from __future__ import annotations

import numpy as np

from progressivis.core.utils import Dialog, indices_len, inter_slice, fix_loc
from progressivis.core.bitmap import bitmap
from progressivis.utils.inspect import filter_kwds
from progressivis.table.nary import NAry, ReturnRunStep
from progressivis.table.table import BaseTable, Table
from progressivis.table.dshape import dshape_join

from typing import List, cast, Dict, Any


def join(
    table: BaseTable,
    other: BaseTable,
    name: str = None,
    on=None,
    how: str = "left",
    lsuffix: str = "",
    rsuffix: str = "",
    sort=False,
) -> Table:
    # pylint: disable=too-many-arguments, invalid-name
    "Compute the join of two table."
    if sort:
        raise ValueError("'sort' not yet implemented in Table.join()")
    if on is not None:
        raise ValueError("'on' not yet implemented in Table.join()")
    dshape, rename = dshape_join(table.dshape, other.dshape, lsuffix, rsuffix)
    join_table = Table(name=name, dshape=dshape)
    if how == "left":
        if np.array_equal(table.index, other.index):
            join_table.resize(len(table), index=table.index)
            left_cols = [rename["left"].get(c, c) for c in table.columns]
            right_cols = [rename["right"].get(c, c) for c in other.columns]
            join_table.loc[:, left_cols] = table.loc[:, table.columns]
            join_table.loc[:, right_cols] = other.loc[:, other.columns]
    else:
        raise ValueError("how={} not yet implemented".format(how))
    return join_table


def join_reset(dialog: Dialog) -> None:
    bag = dialog.bag
    bag.first_orphans = bitmap([])
    bag.second_orphans = bitmap([])
    bag.existing_ids = None


def join_start(
    table: BaseTable,
    other: BaseTable,
    dialog: Dialog,
    name: str = None,
    on=None,
    how: str = "left",
    created=None,
    updated=None,
    deleted=None,
    order=("c", "u", "d"),
    reset=False,
    lsuffix="",
    rsuffix="",
    sort=False,
) -> dict:
    # pylint: disable=too-many-arguments, invalid-name, too-many-locals, unused-argument
    "Start the progressive join function"
    if sort:
        raise ValueError("'sort' not yet implemented in Table.join()")
    if on is not None:
        raise ValueError("'on' not yet implemented in Table.join()")
    dshape, rename = dshape_join(table.dshape, other.dshape, lsuffix, rsuffix)
    left_cols = [rename["left"].get(c, c) for c in table.columns]
    right_cols = [rename["right"].get(c, c) for c in other.columns]
    if how == "left":
        # first, second = table, other
        first_key, second_key = "table", "other"
        first_cols, second_cols = left_cols, right_cols
    elif how == "right":
        # first, second = other, table
        first_key, second_key = "other", "table"
        first_cols, second_cols = right_cols, left_cols
    else:
        raise ValueError("how={} not yet implemented".format(how))

    bag = dialog.bag
    bag.dshape = dshape
    bag.first_cols = first_cols
    bag.second_cols = second_cols
    bag.first_key = first_key
    bag.second_key = second_key
    bag.how = how
    join_reset(dialog)
    join_table = Table(name=name, dshape=dshape)
    dialog.set_output_table(join_table)
    dialog.set_started()
    return join_cont(table, other, dialog, created, updated, deleted, order)


def join_cont(
    table: BaseTable,
    other: BaseTable,
    dialog: Dialog,
    created=None,
    updated=None,
    deleted=None,
    order="cud",
    reset=False,
) -> dict:
    # pylint: disable=too-many-arguments, invalid-name, too-many-locals, unused-argument
    "Continue the progressive join function"
    join_table = dialog.output_table
    first_cols = dialog.bag.first_cols
    second_cols = dialog.bag.second_cols
    first_key = dialog.bag.first_key
    second_key = dialog.bag.second_key
    how = dialog.bag.how
    if how == "left":
        first, second = table, other
    else:
        first, second = other, table
    _len = indices_len
    _fix = fix_loc

    def _void(obj):
        if isinstance(obj, slice) and obj.start == obj.stop:
            return True
        return not obj

    def _process_created_outer(ret):
        pass

    def _process_created(ret):
        b = dialog.bag
        if not created:
            return
        if how == "outer":
            return _process_created_outer(ret)
        # if first_key not in created: return
        first_ids = created.get(first_key, None)
        second_ids = created.get(second_key, None)
        only_1st, common, only_2nd = inter_slice(first_ids, second_ids)
        if first_ids is not None:
            new_size = _len(first_ids)
            if (
                isinstance(first_ids, slice)
                and join_table.is_identity
                and (
                    join_table.last_id + 1 == first_ids.start or join_table.last_id == 0
                )
            ):
                # the nice case (no gaps)
                join_table.resize(new_size)
            else:  # there are gaps ...we have to keep trace of existing ids
                join_table.resize(new_size, index=bitmap.asbitmap(first_ids))
                if b.existing_ids is None:
                    b.existing_ids = bitmap.asbitmap(join_table.index)
                else:
                    b.existing_ids = bitmap.union(
                        b.existing_ids, bitmap.asbitmap(first_ids)
                    )
            join_table.loc[_fix(first_ids), first_cols] = first.loc[
                _fix(first_ids), first.columns
            ]
        if not _void(common):
            join_table.loc[_fix(common), second_cols] = second.loc[
                _fix(common), second.columns
            ]
        # first matching: older orphans on the second table with new orphans on the first
        only_1st_bm = bitmap.asbitmap(only_1st)
        paired = b.second_orphans & only_1st_bm
        if paired:
            join_table.loc[paired, second_cols] = second.loc[paired, second.columns]
            b.second_orphans = b.second_orphans - paired
            only_1st_bm -= paired
        b.first_orphans = bitmap.union(b.first_orphans, only_1st_bm)
        # 2nd matching: older orphans on the first table with new orphans on the second
        only_2nd_bm = bitmap.asbitmap(only_2nd)
        paired = b.first_orphans & only_2nd_bm
        if paired:
            join_table.loc[paired, second_cols] = second.loc[paired, second.columns]
            b.first_orphans = b.first_orphans - paired
            only_2nd_bm -= paired
        b.second_orphans = bitmap.union(b.second_orphans, only_2nd_bm)

    def _process_updated(ret):
        if not updated:
            return
        first_ids = updated.get(first_key, None)
        second_ids = updated.get(second_key, None)
        if first_ids:
            join_table.loc[_fix(first_ids), first_cols] = first.loc[
                _fix(first_ids), first.columns
            ]
        if second_ids:
            if join_table.is_identity:
                xisting_ = slice(0, join_table.last_id + 1, 1)
            else:
                xisting_ = dialog.existing_ids
            _, common, _ = inter_slice(second_ids, xisting_)
            join_table.loc[_fix(common), second_cols] = second.loc[
                _fix(common), second.columns
            ]

    def _process_deleted(ret):
        pass

    order_dict = {"c": _process_created, "u": _process_updated, "d": _process_deleted}
    ret: Dict[Any, Any] = {}
    for operator in order:
        order_dict[operator](ret)
    return ret


class Join(NAry):
    "Module executing join."

    def __init__(self, **kwds):
        """Join(on=None, how='left', lsuffix='', rsuffix='',
                sort=False,name=None)
        """
        super(Join, self).__init__(**kwds)
        self.join_kwds = filter_kwds(kwds, join)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        frames: List[BaseTable] = []
        for name in self.get_input_slot_multiple():
            slot = self.get_input_slot(name)
            table = cast(BaseTable, slot.data())
            slot.clear_buffers()
            frames.append(table)
        table = frames[0]
        for other in frames[1:]:
            table = join(table, other, **self.join_kwds)
        length = len(table)
        if self.result is None:
            self.result = table
        return self._return_run_step(self.state_blocked, steps_run=length)
