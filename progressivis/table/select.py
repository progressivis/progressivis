from __future__ import absolute_import, division, print_function

from ..core.utils import ProgressiveError, indices_len, is_valid_identifier
from ..core.slot import SlotDescriptor
from .module import TableModule
from .table import Table
from ..core.bitmap import bitmap

import logging
logger = logging.getLogger(__name__)

class Select(TableModule):
    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True),
                         SlotDescriptor('select', type=bitmap, required=True)])
        super(Select, self).__init__(**kwds)
        self.default_step_size = 1000
        # dependant modules
        self.input_module = None
        self.input_slot = None
        self.query = None
        self.min = None
        self.max = None
        self.min_value = None
        self.max_value = None
        
    def create_dependent_modules(self, input_module, input_slot, **kwds):
        from .range_query import RangeQuery
        
        if self.input_module is not None:
            return self
        dataflow=self.dataflow
        self.input_module = input_module
        self.input_slot = input_slot

        query = RangeQuery(group=self.name, dataflow=dataflow)
        query.create_dependent_modules(input_module, input_slot, **kwds)

        select = self
        select.input.df = input_module.output[input_slot]
        select.input.query = query.output.query

        self.query = query
        self.min = query.min
        self.max = query.max
        self.min_value = query.min_value
        self.max_value = query.max_value
        
        return select
        
    def run_step(self, run_number, step_size, howlong):
        table_slot = self.get_input_slot('table')
        table = table_slot.data()
        table_slot.update(run_number,
                          buffer_created=False,
                          buffer_updated=True,
                          buffer_deleted=False,
                          manage_columns=False)
        
        select_slot = self.get_input_slot('select')
        select_slot.update(run_number,
                           buffer_created=True,
                           buffer_updated=False,
                           buffer_deleted=True)
                           
        steps = 0
        if self._table is None:
            if self._columns is None:
                dshape = table.dshape
            else:
                cols_dshape = [ "%s: %s"%(col, table[col].dshape) for col in self._columns ]
                dshape = '{' + ",".join(cols_dshape) + '}'
            self._table = Table(self.generate_table_name(table.name),
                                dshape=dshape,
                                create=True)

        if select_slot.deleted.any():
            indices = select_slot.deleted.next(step_size*2, as_slice=False)
            s = indices_len(indices)
            logger.info("deleting %s",indices)
            del self._table.loc[indices]
            steps += s//2
            step_size -= s//2

        if step_size > 0 and select_slot.created.any():
            indices = select_slot.created.next(step_size, as_slice=False)
            s = indices_len(indices)
            logger.info("creating %s",indices)
            steps += s
            step_size -= s
            if self._columns is None:
                #TODO
                # ind = np.array(indices, dtype=np.int64)
                # for column in self._columns:
                #   values = table._column(column)[ind]
                #   self._table._column(i).append(values, indices=ind)
                for i in indices:
                    row = table.row(i)
                    self._table.add(row, index=i)
            else:
                row = { c: None for c in self._columns }
                for i in indices:
                    idx = table.id_to_index(i)
                    for c in self._columns:
                        row[c] = table[c][idx]
                    self._table.add(row, index=i)

        if step_size > 0 and table_slot.updated.any():
            indices = table_slot.updated.next(step_size, as_slice=False)
            logger.info("updating %d", indices)
            s = indices_len(indices)
            steps += s
            step_size -= s
            if self._columns is None:
                for i in indices:
                    self._table.loc[i] = table.loc[i]
            else:
                row = { c: None for c in self._columns }
                for i in indices:
                    for c in self._columns:
                        idx = table.id_to_index(i)
                        row[c] = table[c][idx]
                    self._table.loc[i] = row

        #print('index: ', len(self._table.index))
        return self._return_run_step(self.next_state(select_slot), steps_run=steps)

    @staticmethod
    def make_range_query(column, low, high=None):
        if not is_valid_identifier(column):
            raise ProgressiveError('Cannot use column "%s", invalid name in expression',column)
        if high==None or low==high:
            return "({} == {})".format(low,column)
        elif low > high:
            low,high = high, low
        return "({} <= {} <= {})".format(low,column,high)

    @staticmethod
    def make_and_query(*expr):
        if len(expr)==1:
            return expr[0]
        elif len(expr)>1:
            return " and ".join(expr)
        return ""

    @staticmethod
    def make_or_query(*expr):
        if len(expr)==1:
            return expr[0]
        elif len(expr)>1:
            return " or ".join(expr)
        return ""
