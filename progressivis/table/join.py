"Join Module."
from __future__ import annotations

import numpy as np
import logging
from ..core.utils import nn, integer_types
from ..core.pintset import PIntSet
from ..core.module import Module, ReturnRunStep, def_input, def_output
from .group_by import GroupBy, SubPColumn as SC
from .unique_index import UniqueIndex
from .api import PTable, PTableSelectedView
from .compute import MultiColFunc
from typing import Union, Literal, List, Any, Optional, Dict, Callable


HOW = Union[Literal["inner"], Literal["outer"]]
ON = Optional[Union[str, List[str]]]


def _dt_to_mask(mask: str) -> Any:
    if mask is None:
        return
    return np.array(
        ["Y" in mask, "M" in mask, "D" in mask, "h" in mask, "m" in mask, "s" in mask],
        dtype=int,
    )


def make_ufunc(
    rel_on: Union[str, List[str]],
    ucol: str,
    uindex: Dict[Any, int],
    utable: PTable,
    dtype: np.dtype[Any],
    fillna: Any,
    inv_mask: Any,
    cache: Dict[Any, Any],
) -> Callable[..., Any]:
    inv_mask = _dt_to_mask(inv_mask)
    if isinstance(rel_on, (list, tuple)):

        def _ufunc(ix: Any, local_dict: Dict[Any, Any]) -> Any:
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

        def _ufunc(ix: Any, local_dict: Dict[Any, Any]) -> Any:
            for values in local_dict.values():
                shape_ = values.shape
                break
            if len(shape_) == 1:

                def _cast_inp(x: Any) -> Any:
                    return x

            elif inv_mask is None:

                def _cast_inp(x: Any) -> Any:
                    return tuple(x)

            else:

                def _cast_inp(x: Any) -> Any:
                    return tuple(x * inv_mask)

            res = np.empty(shape_[0], dtype=dtype)
            for i, inp in enumerate(values):
                inp = _cast_inp(inp)
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
            if shape_[0] == 1 and isinstance(ix, integer_types):
                return res[0]
            return res

    return _ufunc


logger = logging.getLogger(__name__)


def _aslist(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return [x]


@def_input("primary", PTable, doc="UniqueIndex output => table contains a primary key")
@def_input("related", PTable, doc=("GroupBy 'result' output => table providing the"
                                   " output index and containing the foreign key"))
@def_output("result", PTableSelectedView)
@def_output(
    "primary_outer", PTableSelectedView, required=False, attr_name="_primary_outer"
)
class Join(Module):
    """
    {many|one}-to-one join module

    Args:
        primary_on: column or list of columns giving the primary key on the primary table
        related_on:  column or list of columns giving the foreign key on the related table
        on: shortcut when primary_on and related_on are identical
        how: {inner|outer} NB: outer provides only outer rows on related table
        kwds : argument to pass to the join function
    """

    def __init__(
        self,
        *,
        how: HOW = "inner",
        fillna: Any = None,
        inv_mask: Any = None,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self.how = how
        self._fillna = fillna
        if nn(inv_mask) and not isinstance(inv_mask, str):
            raise ValueError(f"Mask type {type(inv_mask)} not implemented")
        self._inv_mask = inv_mask
        self._suffix = ""
        self._related_cols: Optional[List[str]] = None
        self._virtual_cols: Optional[List[str]] = None
        self._maintain_primary_outer = False

    def create_dependent_modules(
        self,
        primary_module: Module,
        related_module: Module,
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
            primary_cols: primary table (virtual) columns to be included in the output
                          view
            related_cols: related table columns to be included in the output view
            primary_on: column or list of columns giving the primary key on the primary
                        table
            related_on: column or list of columns giving the foreign key on the related
                        table
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
        if self._inv_mask is None:
            grby = GroupBy(by=self.related_on, scheduler=s)
        elif isinstance(self._inv_mask, str):
            assert isinstance(self.related_on, str)
            grby = GroupBy(
                by=SC(self.related_on).dt[self._inv_mask], keepdims=True, scheduler=s
            )
        else:  # TODO: check the mask type
            assert isinstance(self.related_on, str)
            grby = GroupBy(
                by=SC(self.related_on).ix[self._inv_mask], keepdims=True, scheduler=s
            )
        grby.input.table = related_module.output[related_slot]
        self.input.related = grby.output.result
        uidx = UniqueIndex(on=self.primary_on, scheduler=s)
        uidx.input.table = primary_module.output[primary_slot]
        self.input.primary = uidx.output.result
        self._related_cols = related_cols
        self._virtual_cols = primary_cols
        self.on = related_on
        self.dep.unique_index = uidx
        self.dep.group_by = grby
        self.cache_dict: Optional[Dict[str, Dict[Any, Any]]]

    def starting(self) -> None:
        super().starting()
        opt_slot = self.get_output_slot("primary_outer")
        if opt_slot:
            logger.debug("Maintaining primary outer")
            self._maintain_primary_outer = True

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
            related_cols = (
                self._related_cols if nn(self._related_cols) else related_table.columns
            )
            if set(related_cols) & set(ucols):
                assert self._suffix

            def _sx(x: str) -> str:
                return f"{x}{self._suffix}" if x in related_cols else x

            ucols_dict = {ucol: _sx(ucol) for ucol in ucols}
            self.cache_dict = {c: {} for c in ucols_dict.values()}
            join_cols = list(related_cols) + list(ucols_dict.values())
            assert uindex_mod.result is not None
            computed = {
                sxcol: MultiColFunc(
                    func=make_ufunc(
                        self.on ,
                        ucol,
                        uindex_mod.index,
                        primary_table,
                        uindex_mod.result._column(ucol).dtype,
                        self._fillna,
                        self._inv_mask,
                        self.cache_dict[sxcol],
                    ),
                    base=self.on if isinstance(self.on, list) else [self.on],
                    xshape=uindex_mod.result._column(ucol).shape[1:],
                    dshape=uindex_mod.result._column(ucol).dshape,
                    dtype=uindex_mod.result._column(ucol).dtype,
                )
                for (ucol, sxcol) in ucols_dict.items()
            }
            self.result = PTableSelectedView(
                related_table, PIntSet([]), columns=join_cols, computed=computed
            )
        if self._maintain_primary_outer and self._primary_outer is None:
            self._primary_outer = PTableSelectedView(
                primary_table, PIntSet(primary_table.index)
            )
        steps = 0
        # deletions/updates
        if primary_slot.deleted.any() or primary_slot.updated.any():
            if self.cache_dict:
                for d in self.cache_dict.values():
                    d.clear()
            related_slot.reset()
            return self._return_run_step(self.state_blocked, steps_run=0)
        if related_slot.deleted.any():
            deleted = related_slot.deleted.next(as_slice=False)
            steps = 1
            assert deleted
            self.result.selection -= deleted
        # creations
        if primary_slot.created.any():
            cr = primary_slot.created.next(as_slice=False)
            if self._primary_outer is not None:
                self._primary_outer.selection |= cr
        if related_slot.created.any():
            uindex_terminated = uindex_mod.state == uindex_mod.state_terminated
            if self.how == "inner" or not uindex_terminated:
                created = related_slot.created.all_changes  # type: ignore
                for k, ids in groupby_mod.index.items():
                    if k not in uindex_mod.index:
                        continue
                    common = ids & created
                    if common:
                        steps += 1  # we suppose step != func(len(common))
                        self.result.selection |= common
                        if self._primary_outer is not None:
                            i = uindex_mod.index[k]
                            if i in self._primary_outer.selection:
                                # TODO: check PIntSet.remove()
                                self._primary_outer.selection -= PIntSet([i])
                        created -= common
                        related_slot.created.remove_from_all(common)  # type: ignore
                    if steps >= step_size:
                        break
                else:  # for-else
                    if uindex_terminated and created:
                        related_slot.created.remove_from_all(created)  # type: ignore
            else:  # outer mode or primary table still in process
                created = related_slot.created.next(length=step_size, as_slice=False)
                self.result.selection |= created
                if self._primary_outer is not None:
                    for k, ids in groupby_mod.index.items():
                        if k not in uindex_mod.index:  # or not ids:
                            continue
                        i = uindex_mod.index[k]
                        if i in self._primary_outer.selection:
                            # TODO: check PIntSet.remove()
                            self._primary_outer.selection -= PIntSet([i])

        # currently updates are ignored
        # NB: we assume that the updates do not concern the "join on" columns
        if related_slot.updated.any():
            updates = related_slot.updated.next(as_slice=False)
            steps += len(updates)
        return self._return_run_step(self.next_state(related_slot), steps_run=steps)
