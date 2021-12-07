
from ..core.utils import indices_len, fix_loc, integer_types
from ..core.slot import SlotDescriptor
from ..table.module import TableModule
from ..table.table import Table
from ..utils.psdict import PsDict
from ..core.decorators import process_slot, run_if_any
import numpy as np

import logging
logger = logging.getLogger(__name__)


class Histogram1D(TableModule):
    """
    """
    parameters = [('bins', np.dtype(int), 128),
                  ('delta', np.dtype(float), -5)]
    inputs = [
        SlotDescriptor('table', type=Table, required=True),
        SlotDescriptor('min', type=PsDict, required=True),
        SlotDescriptor('max', type=PsDict, required=True)
    ]
    schema = "{ array: var * int32, min: float64, max: float64, time: int64 }"

    def __init__(self, column, **kwds):
        super(Histogram1D, self).__init__(dataframe_slot='table', **kwds)
        self.tags.add(self.VISUALIZATION_TAG)
        self.column = column
        self.total_read = 0
        self.default_step_size = 1000
        self._histo = None
        self._edges = None
        self._bounds = None
        self._h_cnt = 0
        self.result = Table(self.generate_table_name('Histogram1D'),
                            dshape=Histogram1D.schema,
                            chunks={'array': (16384, 128)},
                            create=True)

    def reset(self):
        self._histo = None
        self._edges = None
        self._bounds = None
        self.total_read = 0
        self._h_cnt = 0
        if self.result:
            self.result.resize(0)

    def is_ready(self):
        if self._bounds and self.get_input_slot('table').created.any():
            return True
        return super(Histogram1D, self).is_ready()

    @process_slot("table", reset_cb="reset")
    @process_slot("min", "max", reset_if=False)
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table
            min_slot = ctx.min
            max_slot = ctx.max
            if not (dfslot.created.any()
                    or min_slot.has_buffered()
                    or max_slot.has_buffered()):
                logger.info('Input buffers empty')
                return self._return_run_step(self.state_blocked, steps_run=0)
            min_slot.clear_buffers()
            max_slot.clear_buffers()
            bounds = self.get_bounds(min_slot, max_slot)
            if bounds is None:
                logger.debug('No bounds yet at run %d', run_number)
                return self._return_run_step(self.state_blocked, steps_run=0)
            bound_min, bound_max = bounds
            if self._bounds is None:
                delta = self.get_delta(*bounds)
                self._bounds = (bound_min - delta, bound_max + delta)
                logger.info("New bounds at run %d: %s",
                            run_number, self._bounds)
            else:
                (old_min, old_max) = self._bounds
                delta = self.get_delta(*bounds)
                if (bound_min < old_min
                    or bound_max > old_max
                    or bound_min > (old_min + delta)
                    or bound_max < (old_max - delta)):
                    self._bounds = (bound_min - delta, bound_max + delta)
                    logger.info('Updated bounds at run %d: %s',
                                run_number, self._bounds)
                    dfslot.reset()
                    dfslot.update(run_number)
                    min_slot.reset()
                    min_slot.update(run_number)
                    max_slot.reset()
                    max_slot.update(run_number)
                    self.reset()
                    return self._return_run_step(self.state_blocked,
                                                 steps_run=0)
            (curr_min, curr_max) = self._bounds
            if curr_min >= curr_max:
                logger.error('Invalid bounds: %s', self._bounds)
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            indices = dfslot.created.next(step_size)  # returns a slice or ...
            steps = indices_len(indices)
            logger.info('Read %d rows', steps)
            self.total_read += steps
            column = input_df[self.column]
            column = column.loc[fix_loc(indices)]
            bins = self._edges if self._edges is not None else self.params.bins
            histo = None
            if len(column) > 0:
                histo, self._edges = np.histogram(column,
                                                  bins=bins,
                                                  range=[curr_min, curr_max],
                                                  density=False)
                self._h_cnt += len(column)
            if self._histo is None:
                self._histo = histo
            elif histo is not None:
                self._histo += histo
            values = {'array': [self._histo],
                      'min': [curr_min],
                      'max': [curr_max],
                      'time': [run_number]}
            self.result['array'].set_shape((self.params.bins,))
            self.result.append(values)
            return self._return_run_step(self.next_state(dfslot),
                                         steps_run=steps)

    def get_bounds(self, min_slot, max_slot):
        min_df = min_slot.data()
        if len(min_df) == 0 and self._bounds is None:
            return None
        min_ = min_df[self.column]
        max_df = max_slot.data()
        if len(max_df) == 0 and self._bounds is None:
            return None
        max_ = max_df[self.column]
        return (min_, max_)

    def get_delta(self, min_, max_):
        delta = self.params['delta']
        extent = max_ - min_
        if delta < 0:
            return extent*delta/-100.0

    def get_histogram(self):
        min_ = self._bounds[0] if self._bounds else None
        max_ = self._bounds[1] if self._bounds else None
        edges = self._edges
        if edges is None:
            edges = []
        elif isinstance(edges, integer_types):
            edges = [edges]
        else:
            edges = edges.tolist()
        return {"edges": edges,
                "values": self._histo.tolist() if self._histo else [],
                "min": min_,
                "max": max_}

    def get_visualization(self):
        return "histogram1d"

    def to_json(self, short=False):
        json = super(Histogram1D, self).to_json(short)
        if short:
            return json
        return self._hist_to_json(json)

    def _hist_to_json(self, json):
        json['histogram'] = self.get_histogram()
        return json
