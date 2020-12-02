
import numpy as np

from progressivis.core.utils import indices_len

from . import Table
from ..core.slot import SlotDescriptor
from .module import TableModule
from ..core.bitmap import bitmap
from .mod_impl import ModuleImpl
from ..io import Variable
from ..stats import Min, Max
from .hist_index import HistogramIndex
from ..utils.psdict import PsDict


def _get_physical_table(t):
    return t if t.base is None else _get_physical_table(t.base)


class _Selection(object):
    def __init__(self, values=None):
        self._values = bitmap([]) if values is None else values

    def update(self, values):
        self._values.update(values)

    def remove(self, values):
        self._values = self._values - bitmap(values)

    def assign(self, values):
        self._values = values
        
    def add(self, values):
        self._values |= values


class RangeQueryImpl(ModuleImpl):
    def __init__(self, column, hist_index, approximate):
        super(RangeQueryImpl, self).__init__()
        self._table = None
        self._column = column
        self.bins = None
        self._hist_index = hist_index
        self._approximate = approximate
        self.result = None
        self.is_started = False

    def resume(self, lower, upper, limit_changed,
               created=None, updated=None, deleted=None):
        if limit_changed:
            new_sel = self._hist_index.range_query(lower, upper,
                                                   approximate=self._approximate)
            self.result.assign(new_sel)
            return
        if updated:
            self.result.remove(updated)
            # res = self._eval_to_ids(limit, updated)
            res = self._hist_index.restricted_range_query(lower, upper,
                                                          only_locs=updated,
                                                          approximate=self._approximate)
            self.result.add(res)
        if created:
            res = self._hist_index.restricted_range_query(lower, upper,
                                                          only_locs=created,
                                                          approximate=self._approximate)
            self.result.update(res)
        if deleted:
            self.result.remove(deleted)

    def start(self, table, lower, upper, limit_changed,
              created=None, updated=None, deleted=None):
        self._table = table
        self.result = _Selection()
        self.is_started = True
        return self.resume(lower, upper, limit_changed,
                           created, updated, deleted)


class RangeQuery(TableModule):
    """
    """
    parameters = [('column', str, "unknown"),
                  ("watched_key_lower", str, ""),
                  ("watched_key_upper", str, ""),
                  # ('hist_index', object, None) # to improve ...
                  ]
    inputs = [SlotDescriptor('table', type=Table, required=True),
              SlotDescriptor('lower', type=Table, required=False),
              SlotDescriptor('upper', type=Table, required=False),
              SlotDescriptor('min', type=PsDict, required=False),
              SlotDescriptor('max', type=PsDict, required=False)]
    outputs = [SlotDescriptor('min', type=Table, required=False),
               SlotDescriptor('max', type=Table, required=False)]

    def __init__(self, hist_index=None, approximate=False, **kwds):
        super(RangeQuery, self).__init__(**kwds)
        self._impl = None  # RangeQueryImpl(self.params.column, hist_index)
        self._hist_index = None
        self._approximate = approximate
        self._column = self.params.column
        self._watched_key_lower = self.params.watched_key_lower
        if not self._watched_key_lower:
            self._watched_key_lower = self._column
        self._watched_key_upper = self.params.watched_key_upper
        if not self._watched_key_upper:
            self._watched_key_upper = self._column
        self.default_step_size = 1000
        self.input_module = None
        self._min_table = None
        self._max_table = None

    @property
    def hist_index(self):
        return self._hist_index

    @hist_index.setter
    def hist_index(self, hi):
        self._hist_index = hi
        self._impl = RangeQueryImpl(self._column, hi,
                                    approximate=self._approximate)

    def create_dependent_modules(self,
                                 input_module,
                                 input_slot,
                                 min_=None,
                                 max_=None,
                                 min_value=None,
                                 max_value=None,
                                 **kwds):
        if self.input_module is not None:  # test if already called
            return self
        scheduler = self.scheduler()
        params = self.params
        self.input_module = input_module
        self.input_slot = input_slot
        with scheduler:
            hist_index = HistogramIndex(column=params.column, group=self.name,
                                        scheduler=scheduler)
            hist_index.input.table = input_module.output[input_slot]
            if min_ is None:
                min_ = Min(group=self.name, columns=[self._column],
                           scheduler=scheduler)
                min_.input.table = hist_index.output.min_out
            if max_ is None:
                max_ = Max(group=self.name, columns=[self._column],
                           scheduler=scheduler)
                max_.input.table = hist_index.output.max_out
            if min_value is None:
                min_value = Variable(group=self.name, scheduler=scheduler)
                min_value.input.like = min_.output.table

            if max_value is None:
                max_value = Variable(group=self.name, scheduler=scheduler)
                max_value.input.like = max_.output.table

            range_query = self
            range_query.hist_index = hist_index
            range_query.input.table = hist_index.output.table
            if min_value:
                range_query.input.lower = min_value.output.table
            if max_value:
                range_query.input.upper = max_value.output.table
            range_query.input.min = min_.output.table
            range_query.input.max = max_.output.table

        self.min = min_
        self.max = max_
        self.min_value = min_value
        self.max_value = max_value
        return range_query

    def _create_min_max(self):
        if self._min_table is None:
            self._min_table = PsDict({self._column: np.inf})
        if self._max_table is None:
            self._max_table = PsDict({self._column: -np.inf})

    def _set_minmax_out(self, attr_, val):
        d = {self._column: val}
        if getattr(self, attr_) is None:
            setattr(self, attr_, PsDict(d))
        else:
            getattr(self, attr_).update(d)

    def _set_min_out(self, val):
        return self._set_minmax_out('_min_table', val)

    def _set_max_out(self, val):
        return self._set_minmax_out('_max_table', val)

    def get_data(self, name):
        if name == 'min':
            return self._min_table
        if name == 'max':
            return self._max_table
        return super(RangeQuery, self).get_data(name)

    def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot('table')
        self._create_min_max()
        #
        # lower/upper
        #
        lower_slot = self.get_input_slot('lower')
        # lower_slot.update(run_number)
        upper_slot = self.get_input_slot('upper')
        limit_changed = False
        if lower_slot.deleted.any():
            lower_slot.deleted.next()
        if lower_slot.updated.any():
            lower_slot.updated.next()
            limit_changed = True
        if lower_slot.created.any():
            lower_slot.created.next()
            limit_changed = True
        if not (lower_slot is upper_slot):
            # upper_slot.update(run_number)
            if upper_slot.deleted.any():
                upper_slot.deleted.next()
            if upper_slot.updated.any():
                upper_slot.updated.next()
                limit_changed = True
            if upper_slot.created.any():
                upper_slot.created.next()
                limit_changed = True
        #
        # min/max
        #
        min_slot = self.get_input_slot('min')
        min_slot.clear_buffers()
        # min_slot.update(run_number)
        # min_slot.created.next()
        # min_slot.updated.next()
        # min_slot.deleted.next()
        max_slot = self.get_input_slot('max')
        max_slot.clear_buffers()
        # max_slot.update(run_number)
        # max_slot.created.next()
        # max_slot.updated.next()
        # max_slot.deleted.next()
        if (lower_slot.data() is None or upper_slot.data() is None
                or len(lower_slot.data()) == 0 or len(upper_slot.data()) == 0):
            return self._return_run_step(self.state_blocked, steps_run=0)
        lower_value = lower_slot.data().get(self._watched_key_lower)
        upper_value = upper_slot.data().get(self._watched_key_upper)
        if lower_slot.data() is None or upper_slot.data() is None \
           or min_slot.data() is None or max_slot.data() is None \
           or len(min_slot.data()) == 0 or len(max_slot.data()) == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        minv = min_slot.data().get(self._watched_key_lower)
        maxv = max_slot.data().get(self._watched_key_upper)
        if lower_value is None or \
           np.isnan(lower_value) or \
           lower_value < minv or \
           lower_value >= maxv:
            lower_value = minv
            limit_changed = True
        if upper_value is None or \
           np.isnan(upper_value) or \
           upper_value > maxv or \
           upper_value <= minv or \
           upper_value <= lower_value:
            upper_value = maxv
            limit_changed = True
        self._set_min_out(lower_value)
        self._set_max_out(upper_value)
        # input_slot.update(run_number)
        if not input_slot.has_buffered() and not limit_changed:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # ...
        steps = 0
        deleted = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(step_size)
            steps += indices_len(deleted)
        created = None
        if input_slot.created.any():
            created = input_slot.created.next(step_size)
            steps += indices_len(created)
        updated = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(step_size)
            steps += indices_len(updated)
        input_table = input_slot.data()
        if not self._table:
            self._table = input_table.loc[bitmap([]), :]
        if not self._impl.is_started:
            self._impl.start(input_table, lower_value, upper_value,
                             limit_changed,
                             created=created,
                             updated=updated,
                             deleted=deleted)
        else:
            self._impl.resume(lower_value, upper_value,
                              limit_changed,
                              created=created,
                              updated=updated,
                              deleted=deleted)
        self._table.index = self._impl.result._values
        return self._return_run_step(self.next_state(input_slot), steps)
