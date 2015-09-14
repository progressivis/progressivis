from progressivis.core.common import ProgressiveError

import numpy as np

import logging
logger = logging.getLogger(__name__)


class TimePredictor(object):
    default = None

    def fit(self, trace_df):
        pass

    def predict(self, duration, default_step):
        pass

    @property
    def id(self):
        return self.get_id()

    @id.setter
    def id(self, n):
        self.set_id(n)

    def get_id(self):
        if getattr(self, '_id'):
            return self._id
        return "<anonymous>"

    def set_id(self, id):
        self._id = id

class ConstantTimePredictor(TimePredictor):
    def __init__(self, t):
        self.t = t

    def predict(self, duration, default_step):
        return self.t

class LinearTimePredictor(TimePredictor):
    def __init__(self):
        self.a = 0
        self.calls = 0

    def fit(self, trace_df):
        self.calls += 1
        step_traces = trace_df[trace_df['type']=='step']
        n = len(step_traces)
        if n < 1:
            return
        if n > 7:
            step_traces = step_traces.iloc[n-7:]
        durations = step_traces['duration']
        operations = step_traces.steps_run #reads + step_traces.updates
        logger.info('LinearPredictor %s: Fitting %s/%s', self.id, operations.values, durations.values)
        num = operations.sum()
        time = durations.sum()
        if num == 0:
            return
        a = num / time
        logger.info('LinearPredictor %s Fit: %f operations per second', self.id, a)
        if a > 0:
            self.a = a
        else:
            logger.debug('LinearPredictor %s: predictor fit found a negative slope, ignoring', self.id);

    def predict(self, duration, default):
        if self.a == 0:
            return default
        #TODO account for the confidence interval and take min of the 95% CI
        steps = int(np.max([0, duration*self.a]))
        logger.debug('LinearPredictor %s: Predicts %d steps for duration %f', self.id, steps, duration)
        return steps
    
if TimePredictor.default is None:
    TimePredictor.default = LinearTimePredictor

