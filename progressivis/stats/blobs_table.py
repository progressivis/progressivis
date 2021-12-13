"""
Isotropic Gaussian blobs
"""

from collections import OrderedDict
import logging

import numpy as np
from abc import ABCMeta, abstractmethod
from ..utils.errors import ProgressiveError, ProgressiveStopIteration
from progressivis import ProgressiveError, SlotDescriptor
from ..table.module import TableModule
from ..table.table import Table
from ..table.constant import Constant
from ..utils.psdict import PsDict
from ..core.utils import integer_types
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle as multi_shuffle

logger = logging.getLogger(__name__)

RESERVOIR_SIZE = 10000

def make_mv_blobs(means, covs, n_samples, **kwds):
    assert len(means) == len(covs)
    n_blobs = len(means)
    size = n_samples // n_blobs
    blobs = []
    labels = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        blobs.append(np.random.multivariate_normal(mean, cov, size, **kwds))
        arr = np.empty(size, dtype='int64')
        arr[:] = i
        labels.append(arr)
    blobs = np.concatenate(blobs)
    labels = np.concatenate(labels)
    return multi_shuffle(blobs, labels)

def xy_to_dict(x, y, i, size, cols):
    res = {}
    k = None if size is None else i + size
    for j, col in enumerate(cols):
        res[col] = x[i:,j] if k is None else x[i:k,j]
    labs = y[i:] if k is None else y[i:k]
    return res, labs

class BlobsTableABC(TableModule):
    """Isotropic Gaussian blobs => table
    The purpose of the "reservoir" approach is to ensure the reproducibility of the results
    """
    outputs = [SlotDescriptor('labels', type=Table, required=False)]
    kw_fun = None
    def __init__(self, columns, rows=-1, dtype='float64', seed=0, throttle=False, **kwds):
        super().__init__(**kwds)
        self.tags.add(self.TAG_SOURCE)
        self._kwds = {} #self._filter_kwds(kwds, self.kw_fun)
        """assert 'centers' in self._kwds
        assert 'n_samples' not in self._kwds
        assert 'n_features' not in self._kwds
        assert 'random_state' not in self._kwds"""
        #self._kwds['n_samples'] = rows
        #self._kwds['n_features'] 
        self.default_step_size = 1000
        if isinstance(columns, integer_types):
            self.columns = ["_%d"%i for i in range(1, columns+1)]
            #self._kwds['n_features'] = columns
        elif isinstance(columns, (list, np.ndarray)):
            self.columns = columns
            #self._kwds['n_features'] = len(columns)
        else:
            raise ProgressiveError('Invalid type for columns')
        self.rows = rows
        self.seed = seed
        self._reservoir = None
        self._labels = None
        self._reservoir_idx = 0
        if throttle and isinstance(throttle, integer_types+(float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        dshape = ", ".join([f"{col}: {dtype}" for col in self.columns])
        dshape = "{" + dshape + "}"
        self.result = Table(self.generate_table_name('table'),
                            dshape=dshape,
                            create=True)
        self.columns = self.result.columns

    def starting(self):
        super().starting()
        opt_slot = self.get_output_slot('labels')
        if opt_slot:
            logger.debug('Maintaining labels')
            self.maintain_labels(True)
        else:
            logger.debug('Not maintaining labels')
            self.maintain_labels(False)

    def maintain_labels(self, yes=True):
        if yes and self._labels is None:
            self._labels = Table(self.generate_table_name('blobs_labels'),
                                 dshape="{labels: int64}",
                                 create=True)
        elif not yes:
            self._labels = None

    def labels(self):
        return self._labels

    def get_data(self, name):
        if name == 'labels':
            return self.labels()
        return super().get_data(name)

    @abstractmethod
    def fill_reservoir(self):
        pass

    def run_step(self, run_number, step_size, howlong):
        if step_size == 0:
            logger.error('Received a step_size of 0')
            return self._return_run_step(self.state_ready, steps_run=0)
        logger.info('generating %d lines', step_size)
        if self.throttle:
            step_size = np.min([self.throttle, step_size])
        if self.rows >= 0 and (len(self.result)+step_size) > self.rows:
            step_size = self.rows - len(self.result)
            logger.info('truncating to %d lines', step_size)
            if step_size <= 0:
                raise ProgressiveStopIteration
        if self._reservoir is None:
            self.fill_reservoir()
        steps = step_size        
        while steps>0:
            level = len(self._reservoir[0]) - self._reservoir_idx
            assert level >=0
            if steps >= level:
                blobs_dict, y_ = xy_to_dict(*self._reservoir, self._reservoir_idx, None, self.columns)
                steps -= level
                # reservoir was emptied so:
                self.fill_reservoir()
            else: # steps < level
                blobs_dict, y_ = xy_to_dict(*self._reservoir, self._reservoir_idx, steps, self.columns)
                self._reservoir_idx += steps
                steps = 0
            self.result.append(blobs_dict)
            if self._labels is not None:
                self._labels.append({'labels': y_})
        if len(self.result) == self.rows:
            next_state = self.state_zombie
        elif self.throttle:
            next_state = self.state_blocked
        else:
            next_state = self.state_ready
        return self._return_run_step(next_state, steps_run=step_size)

class BlobsTable(BlobsTableABC):
    kw_fun = make_blobs
    def __init__(self, *args, **kwds):
        #import pdb;pdb.set_trace()
        super().__init__(*args, **kwds)
        #assert 'centers' in self._kwds
        self.centers =  kwds['centers']
        assert 'n_samples' not in self._kwds
        assert 'n_features' not in self._kwds
        assert 'random_state' not in self._kwds

    def fill_reservoir(self):
        X, y = make_blobs(n_samples=RESERVOIR_SIZE, random_state=self.seed, centers=self.centers, **self._kwds)
        self.seed += 1
        self._reservoir = (X, y)
        self._reservoir_idx = 0

class MVBlobsTable(BlobsTableABC):
    kw_fun = make_mv_blobs
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.means = kwds['means']
        self.covs = kwds['covs']

    def fill_reservoir(self):
        np.random.seed(self.seed)
        X, y = make_mv_blobs(n_samples=RESERVOIR_SIZE, means=self.means, covs=self.covs, **self._kwds)
        self.seed += 1
        self._reservoir = (X, y)
        self._reservoir_idx = 0
    
