import pandas as pd
import numpy as np
import os

class Tracer(object):
    def start_run(self, ts, **kwds):
        pass
    def end_run(self, ts, **kwds):
        pass
    def run_stopped(self, ts, **kwds):
        pass
    def before_run_step(self, ts, **kwds):
        pass
    def after_run_step(self, ts, **kwds):
        pass
    def exception(self, ts, **kwds):
        pass
    def terminated(self, ts, **kwds):
        pass
    def trace_stats(self, max_runs=None):
        return []

class TracerProxy(object):
    def __init__(self, tracer):
        self.tracer = tracer
    def start_run(self, ts, **kwds):
        self.tracer.start_run(ts, **kwds)
    def end_run(self, ts, **kwds):
        self.tracer.end_run(ts, **kwds)
    def run_stopped(self, ts, **kwds):
        self.tracer.run_stopped(ts, **kwds)
    def before_run_step(self, ts, **kwds):
        self.tracer.before_run_step(ts, **kwds)
    def after_run_step(self, ts, **kwds):
        self.tracer.after_run_step(ts, **kwds)
    def exception(self, ts, **kwds):
        self.tracer.exception(ts, **kwds)
    def terminated(self, ts, **kwds):
        self.tracer.terminated(ts, **kwds)
    def trace_stats(self, max_runs=None):
        return self.tracer.trace_stats(max_runs)

class DataFrameTracer(Tracer):
    TRACE_COLUMNS = [
        ('type',      object         , ''),
        ('start',     np.dtype(float), np.nan),
        ('end',       np.dtype(float), np.nan),
        ('duration',  np.dtype(float), np.nan),
        ('detail',    object         , ''),
        ('loadavg',   np.dtype(float), np.nan),
        ('run',       np.dtype(int)  , 0),
        ('step',      np.dtype(int)  , 0),
        ('reads',     np.dtype(int)  , 0),
        ('updates',   np.dtype(int)  , 0),
        ('creates',   np.dtype(int)  , 0),
        ('steps_run', np.dtype(int)  , 0),
        ('next_state',np.dtype(int)  , 0)]

    def __init__(self):
        self.dataframe = None
        self.run_count = 0
        self.step_count = 0
        self.buffer = []
        self.last_run_step_start = None
        self.last_run_step_details = []
        self.last_run_start = None
        self.last_run_details = []

    def df_columns(self):
        return self.TRACE_COLUMNS
        
    def create_df(self):
        if self.dataframe is None:
            columns = {}
            for (name, dtype, dflt) in self.df_columns():
                columns[name] = pd.Series([], dtype=dtype)
            columns['_update'] = columns['end'] # 'end' time becomes '_update' time logically
            del columns['end']
            self.dataframe = pd.DataFrame(columns)
        return self.dataframe

    def df(self):
        if self.dataframe is None:
            self.dataframe = self.create_df()
        if self.buffer:
            columns = self.df_columns()
            d = {name: [] for (name,dtype, dflt) in columns}
            for row in self.buffer:
                for (name,dtype, dflt) in columns:
                    d[name].append(row.get(name, dflt))
            for (name,dtype, dflt) in columns:
                d[name] = pd.Series(d[name], dtype=dtype)
            d['_update'] = d['end'] # 'end' time becomes '_update' time logically
            del d['end']
            df = pd.DataFrame(d)
            self.dataframe = self.dataframe.append(df, ignore_index=True)
            self.buffer = []
        return self.dataframe

    def trace_stats(self, max_runs=None):
        return self.df()

    def start_run(self, ts, **kwds):
        self.last_run_start = {name: dflt for (name,dtype, dflt) in self.df_columns()}
        self.last_run_start['start'] = ts
        self.last_run_start['run'] = self.run_count
        self.step_count = 0

    def end_run(self, ts, **kwds):
        if self.last_run_start is None:
            return
        row = self.last_run_start
        row['end'] = ts
        row['duration'] = ts - row['start']
        row['detail'] = self.last_run_details if self.last_run_details else ''
        row['step'] = self.step_count
        row['loadavg'] = os.getloadavg()[0]
        row['type'] = 'run'
        self.buffer.append(row)
        self.run_count += 1
        self.last_run_details = []
        self.last_run_start = None
        
    def run_stopped(self, ts, **kwds):
        self.last_run_details.append('stopped')

    def before_run_step(self, ts, **kwds):
        self.last_run_step_start = {
            'start': ts,
            'run': self.run_count,
            'step': self.step_count }

    def after_run_step(self, ts, **kwds):
        row = self.last_run_step_start
        last_run_start = self.last_run_start
        row['end'] = ts
        row['duration'] = ts - row['start']
        row['detail'] = self.last_run_step_details if self.last_run_step_details else ''
        for (name,dtype, dflt) in self.df_columns():
            if name not in row:
                row[name] = kwds.get(name, dflt)
        last_run_start['reads'] += row['reads']
        last_run_start['updates'] += row['updates']
        last_run_start['creates'] += row['creates']
        last_run_start['steps_run'] += row['steps_run']
        row['type'] = 'step'
        row['loadavg'] = os.getloadavg()[0]
        self.buffer.append(row)
        self.step_count += 1
        self.last_run_details = []
        self.last_run_step_start = None

    def exception(self, ts, **kwds):
        self.last_run_details.append('exception')

    def terminated(self, ts, **kwds):
        self.last_run_details.append('terminated')

def default_tracer():
    return DataFrameTracer()


TRACE_START_RUN = 'start_run'
TRACE_RUN_STOPPED = 'stop_`run'
TRACE_END_RUN = 'end_run'
TRACE_BEFORE_RUN_STEP = 'before_run_step'
TRACE_AFTER_RUN_STEP  ='after_run_step'
TRACE_EXCEPTION = 'exception'
TRACE_TERMINATED = 'terminated'

class OldTracer(Tracer):
    def __init__(self):
        self._trace = []

    def record_trace(self, timestamp, event, **params):
        params['timestamp'] = timestamp
        params['event'] = event
        self._trace.append(params)

    def start_run(self, ts, **kwds):
        self.record_trace(ts, TRACE_START_RUN, **kwds)
    def end_run(self, ts, **kwds):
        self.record_trace(ts, TRACE_END_RUN, **kwds)
    def run_stopped(self, ts, **kwds):
        self.record_trace(ts, TRACE_RUN_STOPPED, **kwds)
    def before_run_step(self, ts, **kwds):
        self.record_trace(ts, TRACE_BEFORE_RUN_STEP, **kwds)
    def after_run_step(self, ts, **kwds):
        self.record_trace(ts, TRACE_AFTER_RUN_STEP, **kwds)
    def exception(self, ts, **kwds):
        self.record_trace(ts, TRACE_EXCEPTION, **kwds)
    def terminated(self, ts, **kwds):
        self.record_trace(ts, TRACE_TERMINATED, **kwds)

    def trace_stats(self, max_runs=None):
        run_step_traces = []
        run_traces = []
        last_run_step_end = None
        last_run_step_detail = []
        last_run_end = None
        last_run_detail = []
        if not max_runs:
            max_runs = len(self._trace) # upper bound
        for event in reversed(self._trace):
            type=event['event']
            if type==TRACE_START_RUN:
                e = {'subtrace': run_step_traces,
                     'count_subtrace': len(run_step_traces),
                     'start': event['timestamp']}
                if last_run_end:
                    e['end'] = last_run_end['timestamp']
                    e['duration'] = last_run_end['timestamp'] - event['timestamp']
                if last_run_detail:
                    e['details'] = last_run_detail if last_run_detail else ''
                    last_run_detail = []
                run_traces.append(e)
                last_run_end = None
                last_run_detail = []
                run_step_traces = []
                last_run_step_end = None
                if len(run_traces) >= max_runs:
                    break
            elif type==TRACE_RUN_STOPPED:
                last_run_detail.append('stopped')
            elif type==TRACE_END_RUN:
                if last_run_end:
                    last_run_detail.append('missing trace_start_run event')
                last_run_end = event
            elif type==TRACE_BEFORE_RUN_STEP:
                e = {'start': event['timestamp']}
                if last_run_step_end:
                    e['duration'] = last_run_step_end['timestamp'] - event['timestamp']
                    for (k, v) in last_run_step_end.iteritems():
                        if  k == 'timestamp':
                            e['end'] = v
                        elif k != 'event':
                            e[k] = v
                if last_run_step_detail:
                    e['details'] = last_run_step_detail if last_run_step_detail else ''
                    last_run_step_detail = []
                run_step_traces.append(e)
                last_run_step_end = None
            elif type==TRACE_AFTER_RUN_STEP:
                if last_run_step_end:
                    last_run_step_detail.append('missing trace_before_run_step event')
                last_run_step_end = event
            elif type==TRACE_EXCEPTION:
                last_run_detail.append('exception')
            elif type==TRACE_TERMINATED:
                last_run_detail.append('TERMINATED')

        return run_traces
