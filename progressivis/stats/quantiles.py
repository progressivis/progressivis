from __future__ import annotations

import logging

from ..core.decorators import process_slot, run_if_any
from ..table.api import (
    PTableSelectedView, PTable
)
from progressivis.utils.api import PDict
from progressivis.core.api import (
    Module,
    ReturnRunStep,
    def_input,
    def_output,
    document,
    PIntSet,
    indices_len,
    fix_loc,
)
from typing import Dict, Any, List, Set

from datasketches import kll_floats_sketch

logger = logging.getLogger(__name__)


DEBUG = False


@document
@def_input("table", PTable, doc="the input table")
@def_output(
    "result",
    PDict,
    doc=("Return the size processed. Use parametrized slots to access quantiles."),
)
@def_output("table", PTableSelectedView, required=False, doc="Return a view on the input table")
class Quantiles(Module):
    """
    Maintain a sketch data structure to get quantile values for every column
    """

    def __init__(
        self,
        k: int = 200,
        **kwds: Any,
    ) -> None:
        """
        Args:
            columns: columns to be processed. When missing all input columns are processed
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        self.default_step_size = 10000
        self._k = k
        self._klls : List[kll_floats_sketch] = []
        self._cache: Dict[float, PDict] = {}
        self._valid: Set[float] = set()
        self._maintain_table_out = False

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.result.clear()
        if self.table is not None:
            self.table.selection = PIntSet([])
        self._klls = []
        self._cache = {}
        self._valid.clear()

    def starting(self) -> None:
        super().starting()
        opt_slot = self.get_output_slot("table")
        if opt_slot:
            logger.debug("Maintaining output table")
            self._maintain_table_out = True

    def get_data(self, name: str, hint: Any = None) -> Any:
        """Return the data of the named output slot.
        """
        if name == "table":
            return self.table
        if self.result is None:
            return None
        if hint is not None and name == "result":
            # print(f"Getting result with hint {hint}...")
            try:
                quantile = float(hint)
            except ValueError:
                if DEBUG:
                    logger.warning(f"Hint {hint} is not convertible to a float")
                quantile = 0
            if quantile in self._cache:
                if DEBUG:
                    logger.warning(f"returning cached value for {quantile}")
                result = self._cache[quantile]
            else:
                if DEBUG:
                    logger.warning(f"returning non cached value for {quantile}")
                result = PDict()
                self._cache[quantile] = result
            if quantile not in self._valid:
                self._valid.add(quantile)
                if DEBUG:
                    logger.warning(f"Refilling value for {quantile}")
                for i, (k, v) in enumerate(self.result.items()):
                    new_v = self._klls[i].get_quantile(quantile)
                    if v != new_v:
                        result[k] = new_v
                self._cache[quantile] = result
            if DEBUG:
                logger.warning(f"Getting result with hint {hint}: {result}")
            return result
        return super().get_data(name, hint)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            if self._maintain_table_out and self.table is None:
                self.table = PTableSelectedView(ctx.table.data(), PIntSet([]))
            indices = ctx.table.created.next(length=step_size, as_slice=False)
            fixed_indices = fix_loc(indices)
            steps = indices_len(indices)
            df = self.filter_slot_columns(ctx.table, fix_loc(indices))
            if self.result is None:
                columns = [col for col in df.columns if df[col].is_numeric()]
                self.result = PDict({col: 0 for col in columns})
                self._klls = [kll_floats_sketch(self._k) for col in columns]
            for i, k in enumerate(self.result.keys()):
                column = df[k]
                column = column.loc[fixed_indices]
                self._klls[i].update(column)  # type: ignore
                self.result[k] = self._klls[i].n
            self._valid.clear()
            if self._maintain_table_out:
                assert self.table is not None
                self.table.selection |= indices
            return self._return_run_step(self.next_state(ctx.table), steps)
