
from ..core.tracer_base import Tracer
from .table import Table
import numpy as np
import os

class TableTracer(Tracer):
    TRACER_DSHAPE = ('{'
                     "type: string,"
                     "start: real,"
                     "end: real,"
                     "duration: real,"
                     "detail: string,"
                     "loadavg: real,"
                     "run: int64,"
                     "steps: int32,"
                     #"reads: int32,"
                     #"updates: int32,"
                     #"creates: int32,"
                     "steps_run: int32,"
                     "next_state: int32,"
                     "progress_current: real,"
                     "progress_max: real,"
                     "quality: real"
                     '}')
    TRACER_INIT = dict([
        ('type', ''),
        ('start', np.nan),
        ('end', np.nan),
        ('duration', np.nan),
        ('detail', ''),
        ('loadavg', np.nan),
        ('run', 0),
        ('steps', 0),
        #('reads', 0),
        #('updates', 0),
        #('creates', 0),
        ('steps_run', 0),
        ('next_state', 0),
        ('progress_current', 0.0),
        ('progress_max', 0.0),
        ('quality', 0.0)
        ])

    def __init__(self, name, storagegroup):
        self.table = Table('trace_'+name, dshape=TableTracer.TRACER_DSHAPE,
                           storagegroup=storagegroup,
                           chunks=256)
        self.table.add(TableTracer.TRACER_INIT)
        self.step_count = 0
        self.last_run_step_start = None
        self.last_run_step_details = []
        self.last_run_start = None
        self.last_run_details = []

    def trace_stats(self, max_runs=None):
        return self.table

    def start_run(self,ts,run_number,**kwds):
        self.last_run_start = dict(TableTracer.TRACER_INIT)
        self.last_run_start['start'] = ts
        self.last_run_start['run'] = run_number
        self.step_count = 0

    def end_run(self,ts,run_number,**kwds):
        if self.last_run_start is None:
            return
        row = self.last_run_start
        row['end'] = ts
        row['duration'] = ts - row['start']
        row['detail'] = self.last_run_details if self.last_run_details else ''
        row['steps'] = self.step_count
        row['loadavg'] = os.getloadavg()[0]
        row['type'] = 'run'
        row['progress_current'] = kwds.get('progress_current', 0.0)
        row['progress_max'] = kwds.get('progress_max', 0.0)
        row['quality'] = kwds.get('quality', 0.0)
        self.table.add(row)
        self.last_run_details = ''
        self.last_run_start = None
        
    def run_stopped(self,ts,run_number,**kwds):
        self.last_run_details += ('stopped')

    def before_run_step(self,ts,run_number,**kwds):
        self.last_run_step_start = {
            'start': ts,
            'run': run_number,
            'steps': self.step_count }

    def after_run_step(self,ts,run_number,**kwds):
        row = self.last_run_step_start
        last_run_start = self.last_run_start
        for (name, dflt) in TableTracer.TRACER_INIT.items():
            if name not in row:
                row[name] = kwds.get(name, dflt)
        row['end'] = ts
        row['duration'] = ts - row['start']
        row['detail'] = self.last_run_step_details if self.last_run_step_details else ''
        last_run_start['steps_run'] += row['steps_run']
        if 'debug' in kwds:
            row['type'] = 'debug_step'
        else:
            row['type'] = 'step'
        row['loadavg'] = os.getloadavg()[0]
        self.table.add(row)
        self.step_count += 1
        self.last_run_details = ''
        self.last_run_step_start = None

    def exception(self,ts,run_number,**kwds):
        self.last_run_details += ('exception')

    def terminated(self,ts,run_number,**kwds):
        self.last_run_details += ('terminated')

    def get_speed(self, depth=15):
        res = []
        non_zero = self.table.eval('steps_run!=0', as_slice=False)
        sz = min(depth, len(non_zero))
        idx = non_zero[-sz:]
        for d, s in zip(self.table['duration'].loc[idx],
                            self.table['steps_run'].loc[idx]):
            if np.isnan(d) or d==0:
                if not s:
                    continue
                elt = np.nan
                
            else:
                elt = float(s)/d
            if np.isnan(elt):
                elt = None
            res.append(elt)
        return res
        
if Tracer.default is None:
    Tracer.default = TableTracer
