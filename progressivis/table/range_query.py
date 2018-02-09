from ..io import Variable
from ..stats import Min, Max
from .hist_index import HistogramIndex
from .bisectmod import Bisect
from .module import TableModule

class RangeQuery(TableModule):
    parameters = [
        ('column', str, ''),
        ('hist_index', object, None),
    ]

    def __init__(self, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [
                         SlotDescriptor('min_value', type=Table, required=True),
                         SlotDescriptor('max_value', type=Table, required=True)])
        self._min = None
        self._max = None
    def create_dependent_modules(self,
                                    input_module,
                                    input_slot,
                                    min_value=None,
                                    max_value=None,
                                    **kwds):
        
        if hasattr(self, 'input_module'): # test if already called
            return self
        s = self.scheduler()
        p = self.params
        self.input_module = input_module
        self.input_slot = input_slot

        min_ = Min(group=self.id, scheduler=s)
        min_.input.table = input_module.output[input_slot]
        max_ = Max(group=self.id, scheduler=s)
        max_.input.table = input_module.output[input_slot]
        if min_value is None:
            min_value = Variable(group=self.id, scheduler=s)
            min_value.input.like = min_.output.table

        if max_value is None:
            max_value = Variable(group=self.id, scheduler=s)
            max_value.input.like = max_.output.table
        hist_index = HistogramIndex(column=p.column, group=self.id, scheduler=s)
        hist_index.input.table = input_module.output[input_slot]
        hist_index.input.min = min_.output.table
        hist_index.input.max = max_.output.table
        bisect_min = Bisect(column=p.column,op='>', hist_index=hist_index, group=self.id, scheduler=s)
        bisect_min.input.table = hist_index.output.table
        bisect_min.input.limit = min_value.output.table
        bisect_max = Bisect(column=p.column,op='<', hist_index=hist_index, group=self.id, scheduler=s)
        bisect_max.input.table = hist_index.output.table
        bisect_max.input.limit = max_value.output.table
        range_query = self
        range_query.input.min_value = bisect_min.output.table
        range_query.input.max_value = bisect_max.output.table
        return range_query
       
    def run_step(self, run_number, step_size, howlong):
        min_slot = self.get_input_slot('min_value')
        min_slot.update(run_number, self.id)
        max_slot = self.get_input_slot('max_value')
        max_slot.update(run_number, self.id)
        changes = (min_slot.deleted, min_slot.updated, min_slot.created,
                       max_slot.deleted, max_slot.updated, max_slot.created)
        if not any((c.any() for c in changes)):
            return self._return_run_step(self.state_blocked, steps_run=0)
        steps = sum((indices_len(c.next(step_size)) for c in changes))
        min_table = min_slot.data()
        max_table = max_slot.data()
        new_sel = min_table.selection & max_table.selection
        if not self._table:
            self._table = TableSelectedView(min_table, new_sel) # any, actually
        else:
            self._table.selection = new_sel
        return self._return_run_step(self.next_state(min_slot), steps_run=steps)    
            
        
        
                                                    
