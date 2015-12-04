from __future__ import with_statement

import pandas as pd
import numpy as np
import os

class Tracer(object):
    default = None

    def start_run(self,ts,run_number,**kwds):
        pass
    def end_run(self,ts,run_number,**kwds):
        pass
    def run_stopped(self,ts,run_number,**kwds):
        pass
    def before_run_step(self,ts,run_number,**kwds):
        pass
    def after_run_step(self,ts,run_number,**kwds):
        pass
    def exception(self,ts,run_number,**kwds):
        pass
    def terminated(self,ts,run_number,**kwds):
        pass
    def trace_stats(self, max_runs=None):
        return []

class TracerProxy(object):
    def __init__(self, tracer):
        self.tracer = tracer
    def start_run(self,ts,run_number,**kwds):
        self.tracer.start_run(ts,run_number,**kwds)
    def end_run(self,ts,run_number,**kwds):
        self.tracer.end_run(ts,run_number,**kwds)
    def run_stopped(self,ts,run_number,**kwds):
        self.tracer.run_stopped(ts,run_number,**kwds)
    def before_run_step(self,ts,run_number,**kwds):
        self.tracer.before_run_step(ts,run_number,**kwds)
    def after_run_step(self,ts,run_number,**kwds):
        self.tracer.after_run_step(ts,run_number,**kwds)
    def exception(self,ts,run_number,**kwds):
        self.tracer.exception(ts,run_number,**kwds)
    def terminated(self,ts,run_number,**kwds):
        self.tracer.terminated(ts,run_number,**kwds)
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
        self.step_count = 0
        self.buffer = []
        self.last_run_step_start = None
        self.last_run_step_details = []
        self.last_run_start = None
        self.last_run_details = []
        self.columns = [ name for (name, dtype, dflt) in self.df_columns() ]
        self.columns[self.columns.index('run')] = '_update'

    def df_columns(self):
        return self.TRACE_COLUMNS
        
    def create_df(self):
        if self.dataframe is None:
            d = {}
            for (name, dtype, dflt) in self.df_columns():
                d[name] = pd.Series([], dtype=dtype)
            d['_update'] = d['run'] # 'run' number becomes '_update' number
            del d['run']
            self.dataframe = pd.DataFrame(d, columns=self.columns)
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
            d['_update'] = d['run'] # 'run' number becomes '_update' number
            del d['run']
            df = pd.DataFrame(d, columns=self.columns)
            self.dataframe = self.dataframe.append(df, ignore_index=True)
            self.buffer = []
        return self.dataframe

    def trace_stats(self, max_runs=None):
        return self.df()

    def start_run(self,ts,run_number,**kwds):
        self.last_run_start = {name: dflt for (name,dtype, dflt) in self.df_columns()}
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
        row['step'] = self.step_count
        row['loadavg'] = os.getloadavg()[0]
        row['type'] = 'run'
        self.buffer.append(row)
        self.last_run_details = []
        self.last_run_start = None
        
    def run_stopped(self,ts,run_number,**kwds):
        self.last_run_details.append('stopped')

    def before_run_step(self,ts,run_number,**kwds):
        self.last_run_step_start = {
            'start': ts,
            'run': run_number,
            'step': self.step_count }

    def after_run_step(self,ts,run_number,**kwds):
        row = self.last_run_step_start
        last_run_start = self.last_run_start
        for (name,dtype, dflt) in self.df_columns():
            if name not in row:
                row[name] = kwds.get(name, dflt)
        row['end'] = ts
        row['duration'] = ts - row['start']
        row['detail'] = self.last_run_step_details if self.last_run_step_details else ''
        last_run_start['reads'] += row['reads']
        last_run_start['updates'] += row['updates']
        last_run_start['creates'] += row['creates']
        last_run_start['steps_run'] += row['steps_run']
        if 'debug' in kwds:
            row['type'] = 'debug_step'
        else:
            row['type'] = 'step'
        row['loadavg'] = os.getloadavg()[0]
        self.buffer.append(row)
        self.step_count += 1
        self.last_run_details = []
        self.last_run_step_start = None

    def exception(self,ts,run_number,**kwds):
        self.last_run_details.append('exception')

    def terminated(self,ts,run_number,**kwds):
        self.last_run_details.append('terminated')

if Tracer.default is None:
    Tracer.default = DataFrameTracer


