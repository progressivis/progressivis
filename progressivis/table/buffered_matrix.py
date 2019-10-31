
import logging
logger = logging.getLogger(__name__)

import numpy as np

from ..core.utils import next_pow2


class BufferedMatrix(object):
    def __init__(self):
        self._mat = None
        self._base = None

    def reset(self):
        self._mat = None
        self._base = None

    def matrix(self):
        return self._mat

    def allocated_size(self):
        return 0 if self._base is None else self._base.shape[0]

    def __len__(self):
        return 0 if self._mat is None else self._mat.shape[0]
    
    def resize(self, l):
        lb = self.allocated_size()
        if l > lb:
            n = next_pow2(l)
            logger.info('Resizing matrix from %d to %d', lb, n)
            self._base = np.empty([n,n]) # defaults to float
            if self._mat is not None:
                # copy old content
                self._base[0:self._mat.shape[0],0:self._mat.shape[1]] = self._mat
                logger.debug('Base matrix copied')
        self._mat = self._base[0:l,0:l]
        return self._mat
