
from abc import abstractmethod
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TimePredictor(object):
    default = None

    def __init__(self):
        self.name = None

    @abstractmethod
    def fit(self, trace_df):
        pass

    @abstractmethod
    def predict(self, duration, default_step):
        pass


class ConstantTimePredictor(TimePredictor):
    def __init__(self, t):
        super(ConstantTimePredictor, self).__init__()
        self.t = t

    def predict(self, duration, default_step):
        return self.t


class LinearTimePredictor(TimePredictor):
    def __init__(self):
        super(LinearTimePredictor, self).__init__()
        self.a = 0
        self.calls = 0

    def fit(self, trace_df):
        self.calls += 1
        if trace_df is None:
            return

        # TODO optimize to search backward to avoid scanning the whole table
        (step_traces,) = np.where((trace_df['type'] == 'step') &
                                  (trace_df['duration'] != 0))
        n = len(step_traces)
        if n < 1:
            return
        if n > 7:
            step_traces = step_traces[-7:]
        durations = trace_df['duration'][step_traces]
        operations = trace_df['steps_run'][step_traces]
        logger.debug('LinearPredictor %s: Fitting %s/%s',
                     self.name, operations, durations)
        num = operations.sum()
        time = durations.sum()
        if num == 0:
            return
        a = num / time
        logger.info('LinearPredictor %s Fit: %f operations per second',
                    self.name, a)
        if a > 0:
            self.a = a
        else:
            logger.debug('LinearPredictor %s: predictor fit found'
                         ' a negative slope, ignoring', self.name)

    def predict(self, duration, default):
        if self.a == 0:
            return default
        # TODO account for the confidence interval and take min of the 95% CI
        steps = int(np.max([0, np.ceil(duration*self.a)]))
        logger.debug('LinearPredictor %s: Predicts %d steps for duration %f',
                     self.name, steps, duration)
        return steps


if TimePredictor.default is None:
    TimePredictor.default = LinearTimePredictor
