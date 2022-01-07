"""
Visualize a data table with histograms.
"""
from __future__ import annotations

import logging

import numbers
import numpy as np

from progressivis.core.module import JSon
from progressivis.table.nary import NAry
from progressivis.core.slot import SlotDescriptor
from progressivis.stats import Histogram1D
from progressivis.table.table import BaseTable

from typing import cast, Any, Dict, List

logger = logging.getLogger(__name__)


class Histograms(NAry):
    "Visualize a table with multiple histograms"
    parameters = [("bins", np.dtype(int), 128), ("delta", np.dtype(float), -5)]  # 5%
    inputs = [
        SlotDescriptor("min", type=BaseTable, required=True),
        SlotDescriptor("max", type=BaseTable, required=True),
    ]
    outputs = [
        SlotDescriptor("min", type=BaseTable, required=False),
        SlotDescriptor("max", type=BaseTable, required=False),
    ]

    def __init__(self, columns: List[str] = None, **kwds):
        super(Histograms, self).__init__(**kwds)
        self.tags.add(self.TAG_VISUALIZATION)
        self.default_step_size = 1
        self._columns = columns
        self._histogram: Dict[str, Histogram1D] = {}

    def table_(self) -> BaseTable:
        "Return the table"
        return self.get_input_slot("table").data()

    def get_data(self, name: str) -> Any:
        if name == "min":
            return self.get_input_slot("min").data()
        if name == "max":
            return self.get_input_slot("max").data()
        return super(Histograms, self).get_data(name)

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

    def _create_columns(self, columns: List[str], df):
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

    def to_json(self, short=False, with_speed: bool = True) -> JSon:
        json = super(Histograms, self).to_json(short, with_speed)
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
