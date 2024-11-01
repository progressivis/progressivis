from __future__ import annotations

from ..core.module import Module, ReturnRunStep, def_input, def_output
from ..core.pintset import PIntSet
from progressivis.utils.errors import ProgressiveError
from ..core.utils import indices_len, is_valid_identifier
from .table import PTable

from typing import Optional, Any

import logging

logger = logging.getLogger(__name__)


@def_input(
    "table",
    type=PTable,
    required=True,
    buffer_created=False,
    buffer_updated=True,
    buffer_deleted=False,
)
@def_input(
    "select",
    type=PIntSet,
    required=True,
    buffer_created=True,
    buffer_updated=False,
    buffer_deleted=True,
)
@def_output("result", PTable)
class Select(Module):
    """ """

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.default_step_size = 1000
        # dependant modules
        self.input_module: Optional[Module] = None
        self.input_slot: Optional[str] = None
        self.query: Optional[Module] = None
        self.min: Optional[Module] = None
        self.max: Optional[Module] = None
        self.min_value: Optional[Module] = None
        self.max_value: Optional[Module] = None

    def create_dependent_modules(
        self, input_module: Module, input_slot: str, **kwds: Any
    ) -> Select:
        from .range_query import RangeQuery

        if self.input_module is not None:
            return self
        with self.grouped():
            scheduler = self.scheduler
            kwds["scheduler"] = scheduler  # make sure we pass it
            self.input_module = input_module
            self.input_slot = input_slot

            query = RangeQuery(group=self.name, scheduler=scheduler)
            query.create_dependent_modules(input_module, input_slot, **kwds)

            select = self
            select.input.df = input_module.output[input_slot]
            select.input.query = query.output.query

            self.dep.query = query
            self.dep.min = query.dep.min
            self.dep.max = query.dep.max
            self.dep.min_value = query.dep.min_value
            self.dep.max_value = query.dep.max_value
            return select

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        table_slot = self.get_input_slot("table")
        table = table_slot.data()
        # table_slot.update(run_number,
        #                   buffer_created=False,
        #                   buffer_updated=True,
        #                   buffer_deleted=False,
        #                   manage_columns=False)
        select_slot = self.get_input_slot("select")
        # select_slot.update(run_number,
        #                    buffer_created=True,
        #                    buffer_updated=False,
        #                    buffer_deleted=True)
        steps = 0
        cols = table_slot.hint
        if self.result is None:
            if cols is None:
                dshape = table.dshape
            else:
                cols_dshape = [
                    "%s: %s" % (col, table[col].dshape) for col in cols
                ]
                dshape = "{" + ",".join(cols_dshape) + "}"
            self.result = PTable(
                self.generate_table_name(table.name), dshape=dshape, create=True
            )

        assert isinstance(self.result, PTable)
        if select_slot.deleted.any():
            indices = select_slot.deleted.next(length=step_size * 2, as_slice=False)
            s = indices_len(indices)
            logger.info("deleting %s", indices)
            del self.result.loc[indices]
            steps += s // 2
            step_size -= s // 2

        if step_size > 0 and select_slot.created.any():
            indices = select_slot.created.next(length=step_size, as_slice=False)
            assert isinstance(indices, PIntSet)
            s = indices_len(indices)
            logger.info("creating %s", indices)
            steps += s
            step_size -= s
            if cols is None:
                # TODO
                # ind = np.array(indices, dtype=np.int64)
                # for column in self._columns:
                #   values = table._column(column)[ind]
                #   self._table._column(i).append(values, indices=ind)
                for i in indices:
                    row = table.row(i)
                    self.result.add(row, index=i)
            else:
                row = {c: None for c in cols}
                for i in indices:
                    idx = table.id_to_index(i)
                    for c in cols:
                        row[c] = table[c][idx]
                    self.result.add(row, index=i)

        if step_size > 0 and table_slot.updated.any():
            indices = table_slot.updated.next(length=step_size, as_slice=False)
            assert isinstance(indices, PIntSet)
            logger.info("updating %d", indices)
            s = indices_len(indices)
            steps += s
            step_size -= s
            if cols is None:
                for i in indices:
                    self.result.loc[i] = table.loc[i]
            else:
                row = {c: None for c in cols}
                for i in indices:
                    for c in cols:
                        idx = table.id_to_index(i)
                        row[c] = table[c][idx]
                    self.result.loc[i] = row
        return self._return_run_step(self.next_state(select_slot), steps_run=steps)

    @staticmethod
    def make_range_query(column: str, low: float, high: Optional[float] = None) -> str:
        if not is_valid_identifier(column):
            raise ProgressiveError('Cannot use column "%s", invalid name', column)
        if high is None or low == high:
            return "({} == {})".format(low, column)
        elif low > high:
            low, high = high, low
        return f"({low} <= {column} <= {high})"

    @staticmethod
    def make_and_query(*expr: str) -> str:
        if len(expr) == 1:
            return expr[0]
        elif len(expr) > 1:
            return " and ".join(expr)
        return ""

    @staticmethod
    def make_or_query(*expr: str) -> str:
        if len(expr) == 1:
            return expr[0]
        elif len(expr) > 1:
            return " or ".join(expr)
        return ""
