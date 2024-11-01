"""
Visualize a data table with histograms.
"""
from __future__ import annotations

import logging

import numbers
import numpy as np

from progressivis.core.api import Module, ReturnRunStep, JSon, def_input, def_output, def_parameter
from progressivis.stats.api import Histogram1D
from progressivis.table.api import BasePTable

from typing import cast, Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@def_parameter("bins", np.dtype(int), 128)
@def_parameter("delta", np.dtype(float), -5)
@def_input("table", BasePTable)
@def_input("min", BasePTable)
@def_input("max", BasePTable)
@def_output("min", BasePTable, required=False)
@def_output("max", BasePTable, required=False)
class Histograms(Module):
    "Visualize a table with multiple histograms"

    def __init__(self, columns: Optional[List[str]] = None, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.tags.add(self.TAG_VISUALIZATION)
        self.default_step_size = 1
        self._columns = columns
        self._histogram: Dict[str, Histogram1D] = {}

    def table_(self) -> BasePTable:
        "Return the table"
        return cast(BasePTable, self.get_input_slot("table").data())

    def get_data(self, name: str, hint: Any = None) -> Any:
        if name == "min":
            return self.get_input_slot("min").data()
        if name == "max":
            return self.get_input_slot("max").data()
        return super().get_data(name, hint)

    def predict_step_size(self, duration: float) -> int:
        return 1

    # def run_old_step(self, run_number, step_size, howlong):
    #     dfslot = self.get_input_slot('table')
    #     input_df = dfslot.data()
    #     # dfslot.update(run_number)
    #     dfslot.clear_buffers()
    #     col_changes = dfslot.column_changes
    #     if col_changes is not None:
    #         self._create_columns(col_changes.created, input_df)
    #         self._delete_columns(col_changes.deleted)
    #     return self._return_run_step(self.state_blocked, steps_run=1)

    def _create_columns(self, columns: List[str], df: Any) -> None:
        bins: int = cast(int, self.params.bins)
        delta: float = cast(float, self.params.delta)  # crude
        inp = self.get_input_module("table")
        minmod = self.get_input_module("min")
        maxmod = self.get_input_module("max")

        assert inp and minmod and maxmod
        for column in columns:
            logger.debug("Creating histogram1d %s", column)
            dtype = df[column].dtype
            if not np.issubdtype(dtype, numbers.Number):
                # only create histograms for number columns
                continue
            histo = Histogram1D(
                group=self.name,
                column=column,
                bins=bins,
                delta=delta,
                scheduler=self.scheduler,
            )
            histo.input.table = inp.output.result
            histo.input.min = minmod.output.result
            histo.input.max = maxmod.output.result
            self.input.table = histo.output._trace  # will become table.1 ...
            self._histogram[column] = histo

    def _delete_columns(self, columns: List[str]) -> None:
        for column in columns:
            logger.debug("Deleting histogram1d %s", column)
            histo = self._histogram[column]
            del self._histogram[column]
            histo.reset()

    def get_visualization(self) -> str:
        return "histograms"

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        json = super().to_json(short, with_speed)
        if short:
            return json
        return self._histograms_to_json(json)

    def _histograms_to_json(self, json: JSon) -> JSon:
        histo_json = {}
        for (column, value) in self._histogram.items():
            column = str(column)
            histo_json[column] = value.get_histogram()
        json["histograms"] = histo_json
        return json

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:  # pragma no cover
        raise NotImplementedError("run_step not defined")
