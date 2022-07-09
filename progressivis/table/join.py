"Join Module."
from __future__ import annotations

import numpy as np
import logging
from ..core.utils import nn
from ..core.bitmap import bitmap
from ..core.module import ReturnRunStep
from ..core.slot import SlotDescriptor
from .module import TableModule
from .group_by import GroupBy
from .unique_index import UniqueIndex
from . import Table, TableSelectedView
from typing import Union, Literal, List, Any, Optional, Dict


HOW = Union[Literal["inner"], Literal["outer"]]
ON = Optional[Union[str, List[str]]]


def make_ufunc(rel_on, ucol, uindex, utable, dtype, fillna, cache):
    if isinstance(rel_on, (list, tuple)):

        def _ufunc(ix, local_dict):
            for values in local_dict.values():
                shape_0 = values.shape[0]
                break
            res = np.empty(shape_0, dtype=dtype)
            all_values = zip(*local_dict.values())
            for i, inp in enumerate(all_values):
                try:
                    val = cache[inp]
                except KeyError:
                    indx = uindex.get(inp)
                    if indx is None:
                        val = fillna
                    else:
                        val = utable.at[indx, ucol]
                    cache[inp] = val
                res[i] = val
            return res

    else:

        def _ufunc(ix, local_dict):
            for values in local_dict.values():
                shape_0 = values.shape[0]
                break
            res = np.empty(shape_0, dtype=dtype)
            for i, inp in enumerate(values):
                try:
                    val = cache[inp]
                except KeyError:
                    indx = uindex.get(inp)
                    if indx is None:
                        val = fillna
                    else:
                        val = utable.at[indx, ucol]
                    cache[inp] = val
                res[i] = val
            return res

    return _ufunc


logger = logging.getLogger(__name__)


def _aslist(x) -> List[Any]:
    if isinstance(x, list):
        return x
    return [x]


class Join(TableModule):
    """
    {many|one}-to-one join module

    Slots:
        primary: UniqueIndex output => table contains a primary key
        related: GroupBy output => table providing the output index and containing the foreign key
    Args:
        primary_on: column or list of columns giving the primary key on the primary table
        related_on:  column or list of columns giving the foreign key on the related table
        on: shortcut when primary_on and related_on are identical
        how: {inner|outer} NB: outer provides only outer rows on related table
        kwds : argument to pass to the join function
    """

    inputs = [
        SlotDescriptor("related", type=Table, required=True),
        SlotDescriptor("primary", type=Table, required=True),
    ]
    outputs = [SlotDescriptor("primary_outer", type=Table, required=False)]

    def __init__(self, *, how: HOW = "inner", fillna: Any = None, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.how = how
        self._fillna = fillna
        self._suffix = ""
        self._related_cols: Optional[List[str]] = None
        self._virtual_cols: Optional[List[str]] = None
        self._maintain_primary_outer = False
        self._primary_outer = None

    def create_dependent_modules(
        self,
        primary_module: TableModule,
        related_module: TableModule,
        *,
        primary_slot: str = "result",
        related_slot: str = "result",
        primary_cols: Optional[List[str]] = None,
        related_cols: Optional[List[str]] = None,
        on: ON = None,
        primary_on: ON = None,
        related_on: ON = None,
        suffix: str = "",
    ) -> None:
        """
        Args:
            primary_module: module providing the primary data source (primary key owner)
            related_module: module providing the related data source (foreign key owner)
            primary_slot: ...
            related_slot: ...
            primary_cols: primary table (virtual) columns to be included in the output view
            related_cols: related table columns to be included in the output view
            primary_on: column or list of columns giving the primary key on the primary table
            related_on:  column or list of columns giving the foreign key on the related table
            on: shortcut when primary_on and related_on are identical
            suffix: ...
        """

        s = self.scheduler()
        on_conf = (nn(on), nn(related_on), nn(primary_on))
        if on_conf not in [(True, False, False), (False, True, True)]:
            raise ValueError(
                "Invalid combination of 'on', 'primary_on' and 'related_on'"
            )
        if nn(on):
            related_on = primary_on = on
        self.related_on = related_on
        self.primary_on = primary_on
        self._suffix = suffix
        assert self.primary_on is not None and self.related_on is not None
        grby = GroupBy(by=related_on, scheduler=s)
        grby.input.table = related_module.output[related_slot]
        self.input.related = grby.output.result
        uidx = UniqueIndex(on=primary_on, scheduler=s)
        uidx.input.table = primary_module.output[primary_slot]
        self.input.primary = uidx.output.result
        self._related_cols = related_cols
        self._virtual_cols = primary_cols
        self.on = related_on
        self.unique_index = uidx
        self.group_by = grby
        self.cache_dict: Optional[Dict[str, Dict[Any, Any]]]

    def starting(self) -> None:
        super().starting()
        opt_slot = self.get_output_slot("primary_outer")
        if opt_slot:
            logger.debug("Maintaining primary outer")
            self._maintain_primary_outer = True

    def get_data(self, name: str) -> Any:
        if name == "primary_outer":
            return self._primary_outer
        return super().get_data(name)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if self.on is None:
            raise ValueError(
                "'on' parameter is not set."
                " Consider running create_dependent_modules() before"
            )
        related_slot = self.get_input_slot("related")
        primary_slot = self.get_input_slot("primary")
        if related_slot is None or primary_slot is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        related_table = related_slot.data()
        primary_table = primary_slot.data()
        if related_table is None or primary_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        groupby_mod = related_slot.output_module
        assert isinstance(groupby_mod, GroupBy)
        uindex_mod = primary_slot.output_module
        assert isinstance(uindex_mod, UniqueIndex)
        if self.result is None:
            ucols = self._virtual_cols or primary_table.columns
            ucols = [uc for uc in ucols if uc not in _aslist(uindex_mod.on)]
            related_cols = self._related_cols or related_table.columns
            if set(related_cols) & set(ucols):
                assert self._suffix

            def _sx(x):
                return f"{x}{self._suffix}" if x in related_cols else x

            ucols_dict = {ucol: _sx(ucol) for ucol in ucols}
            self.cache_dict = {c: {} for c in ucols_dict.values()}
            join_cols = related_cols + list(ucols_dict.values())
            computed = {
                sxcol: dict(
                    vfunc=make_ufunc(
                        self.on,
                        ucol,
                        uindex_mod.index,
                        primary_table,
                        uindex_mod.table._column(ucol).dtype,
                        self._fillna,
                        self.cache_dict[sxcol],
                    ),
                    category="vfunc",
                    cols=self.on,
                    shape=None,
                    dshape=None,
                    dtype=object,
                )
                for (ucol, sxcol) in ucols_dict.items()
            }
            self.result = TableSelectedView(
                related_table, bitmap([]), columns=join_cols, computed=computed
            )

        if self._maintain_primary_outer and self._primary_outer is None:
            self._primary_outer = TableSelectedView(
                primary_table, bitmap(primary_table.index)
            )
        steps = 0
        if related_slot.deleted.any():
            deleted = related_slot.deleted.next(as_slice=False)
            steps = 1
            if deleted:
                self.selected.selection -= deleted
        if primary_slot.deleted.any():
            for d in self.cache_dict.values():
                d.clear()
            deleted = related_slot.deleted.next(as_slice=False)
            if self.how == "inner":
                steps = 1
                for key in uindex_mod.get_deleted_entries(deleted):
                    deltd = groupby_mod.index.get(key, bitmap())
                    if deltd:
                        self.selected.selection -= deltd
        if primary_slot.created.any():
            cr = primary_slot.created.next(as_slice=False)
            if nn(self._primary_outer):
                self._primary_outer.selection |= cr
        if related_slot.created.any():
            uindex_terminated = uindex_mod.state == uindex_mod.state_terminated
            if self.how == "inner" or not uindex_terminated:
                created = related_slot.created.all_changes
                for k, ids in groupby_mod.index.items():
                    if k not in uindex_mod.index:
                        continue
                    common = ids & created
                    if common:
                        steps += len(common)
                        self.selected.selection |= common
                        if nn(self._primary_outer):
                            i = uindex_mod.index[k]
                            if i in self._primary_outer.selection:
                                # TODO: check bitmap.remove()
                                self._primary_outer.selection -= bitmap([i])
                        created -= common
                        related_slot.created.remove_from_all(common)
                    if steps >= step_size:
                        break
                if uindex_terminated and created:
                    related_slot.created.remove_from_all(created)
            else:  # outer mode or primary table still in process
                created = related_slot.created.next(length=step_size, as_slice=False)
                self.selected.selection |= created
                if nn(self._primary_outer):
                    for k, ids in groupby_mod.index.items():
                        if k not in uindex_mod.index:  # or not ids:
                            continue
                        i = uindex_mod.index[k]
                        if i in self._primary_outer.selection:
                            # TODO: check bitmap.remove()
                            self._primary_outer.selection -= bitmap([i])

        # currently updates are ignored
        # NB: we assume that the updates do not concern the "join on" columns
        related_slot.updated.next(as_slice=False)  # nop
        primary_slot.updated.next(as_slice=False)  # nop
        return self._return_run_step(self.next_state(related_slot), steps_run=steps)
