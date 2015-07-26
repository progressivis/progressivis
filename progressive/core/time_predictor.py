from progressive.core.common import ProgressiveError

import numpy as np
from sklearn import linear_model

class TimePredictor(object):
    def fit(self, trace_df):
        pass

    def predict(self, duration, default_step):
        pass

class LinearTimePredictor(TimePredictor):
    def __init__(self):
        self.a = 0
        self.b = 0
        self.calls = 0
        self.lm = linear_model.LinearRegression()

    def fit(self, trace_df):
        self.calls += 1
        step_traces = trace_df[trace_df['type']=='step']
        n = len(step_traces)
        if n < 1:
            return
        if n > 7:
            # limit memory of fit
            last_run = trace_df['run'].irow(-1)
            set_traces = step_traces[step_traces['run'] >= (last_run-1)]
        durations = step_traces['duration']
        operations = step_traces.reads + step_traces.updates
        if n == 1: # be reactive in case default is not conservative enough
            self.a = durations.irow(-1) / operations.irow(-1)
            self.b = 0
        else:
            self.lm.fit(operations[:, np.newaxis], durations)
            self.a = self.lm.coef_[0]
            self.b = self.lm.intercept_

    def predict(self, duration, default):
        if self.a == 0:
            return default
        #TODO account for the confidence interval and take min of the 95% CI
        return np.floor(np.max([0, (duration - self.b) / self.a]))
    
def default_predictor():
    return LinearTimePredictor()
