"""
Range Query module.


"""
from progressivis.core.utils import indices_len
from ..io import Variable
from ..stats import Min, Max
from .hist_index import HistogramIndex
from .bisectmod import Bisect, _get_physical_table
from .module import TableModule
from ..core.slot import SlotDescriptor
from . import Table
from . import TableSelectedView
from ..core.bitmap import bitmap


class RangeQuery(TableModule):
    "Range Query Module"
    parameters = [
        ('column', str, ''),
        ('hist_index', object, None),
    ]

    def __init__(self, scheduler=None, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('min_value', type=Table, required=True),
                         SlotDescriptor('max_value', type=Table, required=True)])
        self._min = None
        self._max = None
        super(RangeQuery, self).__init__(scheduler=scheduler, **kwds)
        self.input_module = None
        self.input_slot = None
        self.min = None
        self.max = None
        self.min_value = None
        self.max_value = None

    def create_dependent_modules(self,
                                 input_module,
                                 input_slot,
                                 min_=None,
                                 max_=None,
                                 min_value=None,
                                 max_value=None,
                                 **kwds):
        if self.input_module is not None: # test if already called
            return self
        s = self.scheduler()
        params = self.params
        self.input_module = input_module
        self.input_slot = input_slot
        if min_ is None:
            min_ = Min(group=self.id, scheduler=s)
            min_.input.table = input_module.output[input_slot]
        if max_ is None:
            max_ = Max(group=self.id, scheduler=s)
            max_.input.table = input_module.output[input_slot]
        if min_value is None:
            min_value = Variable(group=self.id, scheduler=s)
            min_value.input.like = min_.output.table

        if max_value is None:
            max_value = Variable(group=self.id, scheduler=s)
            max_value.input.like = max_.output.table
        hist_index = HistogramIndex(column=params.column, group=self.id, scheduler=s)
        hist_index.input.table = input_module.output[input_slot]
        hist_index.input.min = min_.output.table
        hist_index.input.max = max_.output.table
        bisect_min = Bisect(column=params.column, limit_key=params.column,
                            op='>',
                            hist_index=hist_index,
                            group=self.id,
                            scheduler=s)
        bisect_min.input.table = hist_index.output.table
        bisect_min.input.limit = min_value.output.table
        bisect_max = Bisect(column=params.column, limit_key=params.column,
                            op='<',
                            hist_index=hist_index,
                            group=self.id,
                            scheduler=s)
        bisect_max.input.table = hist_index.output.table
        bisect_max.input.limit = max_value.output.table
        range_query = self
        range_query.input.min_value = bisect_min.output.table
        range_query.input.max_value = bisect_max.output.table
        self.min = min_
        self.max = max_
        self.min_value = min_value
        self.max_value = max_value
        return range_query

    def run_step(self, run_number, step_size, howlong):
        min_slot = self.get_input_slot('min_value')
        min_slot.update(run_number)
        max_slot = self.get_input_slot('max_value')
        max_slot.update(run_number)
        steps = 0 
        min_table = min_slot.data()
        max_table = max_slot.data()
        # min
        deleted_min = None
        if min_slot.deleted.any():
            deleted_min = min_slot.deleted.next(step_size)
            steps += indices_len(deleted_min)
        created_min = None
        if min_slot.created.any():
            created_min = min_slot.created.next(step_size)
            steps += indices_len(created_min)
        updated_min = None
        if min_slot.updated.any():
            updated_min = min_slot.updated.next(step_size)
            steps += indices_len(updated_min)
        # max
        deleted_max = None
        if max_slot.deleted.any():
            deleted_max = max_slot.deleted.next(step_size)
            steps += indices_len(deleted_max)
        created_max = None
        if max_slot.created.any():
            created_max = max_slot.created.next(step_size)
            steps += indices_len(created_max)
        updated_max = None
        if max_slot.updated.any():
            updated_max = max_slot.updated.next(step_size)
            steps += indices_len(updated_max)
        if not self._table:
            self._table = TableSelectedView(_get_physical_table(min_table), bitmap([]))
            new_sel = min_table.selection & max_table.selection
            self._table.selection = new_sel
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        to_remove = (bitmap(deleted_min)|bitmap(deleted_max)|
                     bitmap(updated_min)|bitmap(updated_max))
        to_add = ((bitmap(created_min)|bitmap(updated_min)) &
                  (bitmap(created_max)|bitmap(updated_max)))
        self._table.selection = (self._table.selection - to_remove) | to_add
        return self._return_run_step(self.next_state(min_slot), steps_run=steps)
