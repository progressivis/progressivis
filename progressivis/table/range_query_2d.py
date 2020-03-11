import numpy as np

import itertools as it
from . import Table
from . import TableSelectedView
from ..core.slot import SlotDescriptor
from .module import TableModule
from ..core.bitmap import bitmap
from .mod_impl import ModuleImpl
from ..io import Variable
from ..stats import Min, Max
from .hist_index import HistogramIndex
from progressivis.core.utils import indices_len
from progressivis.utils.synchronized import synchronized
#from progressivis.table.paste import Paste
from ..utils.psdict import PsDict
from ..table.merge_dict import MergeDict

class _Selection(object):
    def __init__(self, values=None):
        self._values = bitmap([]) if values is None else values

    def update(self, values):
        self._values.update(values)

    def remove(self, values):
        self._values = self._values - bitmap(values)

    def assign(self, values):
        self._values = values


class RangeQuery2dImpl(ModuleImpl):
    def __init__(self, column_x, column_y, hist_index_x, hist_index_y,
                 approximate):
        super(RangeQuery2dImpl, self).__init__()
        self._table = None
        self._column_x = column_x
        self._column_y = column_y
        self.bins = None
        self._hist_index_x = hist_index_x
        self._hist_index_y = hist_index_y
        self._approximate = approximate
        self.result = None
        self.is_started = False

    def resume(self, lower_x, upper_x, lower_y, upper_y, limit_changed,
               created=None, updated=None, deleted=None):
        if limit_changed:
            new_sel_x = self._hist_index_x.range_query_aslist(lower_x, upper_x,
                                                approximate=self._approximate)
            new_sel_y = self._hist_index_y.range_query_aslist(lower_y, upper_y,
                                                approximate=self._approximate)
            if new_sel_x is None or new_sel_y is None:
                new_sel_x = self._hist_index_x.range_query(lower_x, upper_x,
                                                approximate=self._approximate)
                new_sel_y = self._hist_index_y.range_query(lower_y, upper_y,
                                                approximate=self._approximate)
                new_sel = new_sel_x & new_sel_y
            else:
                new_sel = bitmap.union(*(x&y for x,y in it.product(new_sel_x, new_sel_y)))
            self.result.assign(new_sel)
            return
        if updated:
            self.result.remove(updated)
            res_x = self._hist_index_x.restricted_range_query(lower_x, upper_x,
                            only_locs=updated, approximate=self._approximate)
            res_y = self._hist_index_y.restricted_range_query(lower_y, upper_y,
                            only_locs=updated, approximate=self._approximate)
            self.result.add(res_x&res_y) # add not defined???
        if created:
            res_x = self._hist_index_x.restricted_range_query(lower_x, upper_x,
                            only_locs=created, approximate=self._approximate)
            res_y = self._hist_index_y.restricted_range_query(lower_y, upper_y,
                            only_locs=created, approximate=self._approximate)
            self.result.update(res_x&res_y)
        if deleted:
            self.result.remove(deleted)

    def start(self, table, lower_x, upper_x, lower_y, upper_y, limit_changed, created=None, updated=None, deleted=None):
        self._table = table
        self.result = _Selection()
        self.is_started = True
        return self.resume(lower_x, upper_x, lower_y, upper_y, limit_changed, created, updated, deleted)


class RangeQuery2d(TableModule):
    """
    """
    parameters = [('column_x', str, "unknown"),
                      ('column_y', str, "unknown"),
                  ("watched_key_lower_x", str, ""),
                  ("watched_key_upper_x", str, ""),                  
                  ("watched_key_lower_y", str, ""),
                  ("watched_key_upper_y", str, ""),                  
                  #('hist_index', object, None) # to improve ...
                 ]

    def __init__(self, hist_index=None, approximate=False, **kwds):
        """
        """
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True),
                         SlotDescriptor('lower', type=Table, required=False),
                         SlotDescriptor('upper', type=Table, required=False),
                         SlotDescriptor('min', type=PsDict, required=False),
                         SlotDescriptor('max', type=PsDict, required=False)
                        ])
        self._add_slots(kwds,'output_descriptors', [
            SlotDescriptor('min', type=Table, required=False),
            SlotDescriptor('max', type=Table, required=False)])
        super(RangeQuery2d, self).__init__(**kwds)
        self._impl = None #RangeQueryImpl(self.params.column, hist_index)
        self._hist_index_x = None
        self._hist_index_y = None        
        self._approximate = approximate
        self._column_x = self.params.column_x
        self._column_y = self.params.column_y
        # X ...
        self._watched_key_lower_x = self.params.watched_key_lower_x
        if not self._watched_key_lower_x:
            self._watched_key_lower_x = self._column_x
        self._watched_key_upper_x = self.params.watched_key_upper_x
        if not self._watched_key_upper_x:
            self._watched_key_upper_x = self._column_x
        # Y ...
        self._watched_key_lower_y = self.params.watched_key_lower_y
        if not self._watched_key_lower_y:
            self._watched_key_lower_y = self._column_y
        self._watched_key_upper_y = self.params.watched_key_upper_y
        if not self._watched_key_upper_y:
            self._watched_key_upper_y = self._column_y
        self.default_step_size = 1000
        self.input_module = None
        self._min_table = None
        self._max_table = None
    #@property
    #def hist_index(self):
    #    return self._hist_index
    #@hist_index.setter
    #def hist_index(self, hi):
    #    self._hist_index = hi
    #    self._impl = RangeQuery2dImpl(self._column, hi, approximate=self._approximate)
    #    raise AttributeError("hist_index not implemented")
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
            hist_index_x = HistogramIndex(column=params.column_x,
                                          group=self.name,
                                          scheduler=scheduler)
            hist_index_x.input.table = input_module.output[input_slot]
            hist_index_y = HistogramIndex(column=params.column_y,
                                          group=self.name,
                                          scheduler=scheduler)
            hist_index_y.input.table = input_module.output[input_slot]
            if min_ is None:
                min_x = Min(group=self.name,
                            scheduler=scheduler,
                            columns=[self._column_x])
                min_x.input.table = hist_index_x.output.min_out
                min_y = Min(group=self.name,
                            scheduler=scheduler,
                            columns=[self._column_y])
                min_y.input.table = hist_index_y.output.min_out
                min_ = MergeDict(group=self.name, scheduler=scheduler)
                min_.input.first = min_x.output.table
                min_.input.second = min_y.output.table
            if max_ is None:
                max_x = Max(group=self.name,
                            scheduler=scheduler,
                            columns=[self._column_x])
                max_x.input.table = hist_index_x.output.max_out
                max_y = Max(group=self.name,
                            scheduler=scheduler,
                            columns=[self._column_y])
                max_y.input.table = hist_index_y.output.max_out
                max_ = MergeDict(group=self.name, scheduler=scheduler)
                max_.input.first = max_x.output.table
                max_.input.second = max_y.output.table
            if min_value is None:
                min_value = Variable(group=self.name,
                                     scheduler=scheduler)
                min_value.input.like = min_.output.table
            if max_value is None:
                max_value = Variable(group=self.name,
                                     scheduler=scheduler)
                max_value.input.like = max_.output.table

            range_query = self
            # range_query.hist_index = hist_index
            self._impl = RangeQuery2dImpl(self._column_x, self._column_y,
                                          hist_index_x, hist_index_y,
                                          approximate=self._approximate)
            range_query.input.table = hist_index_x.output.table  # don't care
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

    @property
    def min_max_dshape(self):
        return '{%s: float64, %s: float64}' % (self._column_x, self._column_y)

    def _create_min_max(self):
        if self._min_table is None:
            self._min_table = Table(name=None, dshape=self.min_max_dshape)
        if self._max_table is None:
            self._max_table = Table(name=None, dshape=self.min_max_dshape)

    def _set_min_out(self, val_x, val_y):
        if self._min_table is None:
            self._min_table = Table(name=None, dshape=self.min_max_dshape)
        if len(self._min_table) == 0:
            self._min_table.append({self._column_x: val_x,
                                    self._column_y: val_y}, indices=[0])
            return
        if self._min_table.last(self._column_x) == val_x and \
           self._min_table.last(self._column_y) == val_y:
            return
        self._min_table[self._column_x].loc[0] = val_x
        self._min_table[self._column_y].loc[0] = val_y

    def _set_max_out(self, val_x, val_y):
        if self._max_table is None:
            self._max_table = Table(name=None, dshape=self.min_max_dshape)
        if len(self._max_table) == 0:
            self._max_table.append({self._column_x: val_x,
                                    self._column_y: val_y}, indices=[0])
            return
        if self._max_table.last(self._column_x) == val_x and \
           self._max_table.last(self._column_y) == val_y:
            return
        self._max_table[self._column_x].loc[0] = val_x
        self._max_table[self._column_y].loc[0] = val_y

    def get_data(self, name):
        if name == 'min':
            return self._min_table
        if name == 'max':
            return self._max_table
        return super(RangeQuery2d, self).get_data(name)

    async def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot('table')
        input_slot.update(run_number)
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
        #with input_slot.lock:
        input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if not self._table:
            self._table = TableSelectedView(input_table, bitmap([]))
        self._create_min_max()
        # param = self.params
        #
        # lower/upper
        #
        lower_slot = self.get_input_slot('lower')
        lower_slot.update(run_number)
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
            upper_slot.update(run_number)
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
        min_slot.update(run_number)
        min_slot.created.next()
        min_slot.updated.next()
        min_slot.deleted.next()        
        max_slot = self.get_input_slot('max')
        max_slot.update(run_number)
        max_slot.created.next()
        max_slot.updated.next()
        max_slot.deleted.next()
        if (lower_slot.data() is None or upper_slot.data() is None
                or len(lower_slot.data()) == 0 or len(upper_slot.data()) == 0):
            return self._return_run_step(self.state_blocked, steps_run=0)
        # X ...
        lower_value_x = lower_slot.data().last(self._watched_key_lower_x)
        upper_value_x = upper_slot.data().last(self._watched_key_upper_x)
        # Y ...
        lower_value_y = lower_slot.data().last(self._watched_key_lower_y)
        upper_value_y = upper_slot.data().last(self._watched_key_upper_y)
        if (lower_slot.data() is None or upper_slot.data() is None
                or len(min_slot.data()) == 0 or len(max_slot.data()) == 0):
            return self._return_run_step(self.state_blocked, steps_run=0)
        # X ...
        minv_x = min_slot.data().get(self._watched_key_lower_x)
        maxv_x = max_slot.data().get(self._watched_key_upper_x)
        # Y ...
        minv_y = min_slot.data().get(self._watched_key_lower_y)
        maxv_y = max_slot.data().get(self._watched_key_upper_y)
        # X ...
        if lower_value_x is None or np.isnan(lower_value_x) or lower_value_x < minv_x or lower_value_x>=maxv_x:
            lower_value_x = minv_x
            limit_changed = True
        if (upper_value_x is None or np.isnan(upper_value_x) or upper_value_x > maxv_x or upper_value_x<=minv_x
                or upper_value_x<=lower_value_x):
            upper_value_x = maxv_x
            limit_changed = True
        # Y ...
        if lower_value_y is None or np.isnan(lower_value_y) or lower_value_y < minv_y or lower_value_y>=maxv_y:
            lower_value_y = minv_y
            limit_changed = True
        if (upper_value_y is None or np.isnan(upper_value_y) or upper_value_y > maxv_y or upper_value_y<=minv_y
                or upper_value_y<=lower_value_y):
            upper_value_y = maxv_y
            limit_changed = True
        self._set_min_out(lower_value_x, lower_value_y)
        self._set_max_out(upper_value_x, upper_value_y)
        if steps==0 and not limit_changed:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # ...
        if not self._impl.is_started:
            status = self._impl.start(input_table, lower_value_x,
                                      upper_value_x, lower_value_y,
                                      upper_value_y, limit_changed,
                                      created=created,
                                      updated=updated,
                                      deleted=deleted)
            self._table.selection = self._impl.result._values
        else:
            status = self._impl.resume(lower_value_x, upper_value_x,
                                       lower_value_y, upper_value_y,
                                       limit_changed,
                                       created=created,
                                       updated=updated,
                                       deleted=deleted)
            self._table.selection = self._impl.result._values
        return self._return_run_step(self.next_state(input_slot), steps)
