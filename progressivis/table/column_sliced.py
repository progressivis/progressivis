
from .column_proxy import ColumnProxy
from progressivis.core.utils import integer_types, norm_slice

import numpy as np
from collections import Iterable

import logging
logger = logging.getLogger(__name__)


class ColumnSlicedView(ColumnProxy):
    def __init__(self, name, base, index, view_slice):
        if not (isinstance(view_slice, slice) and view_slice.step in (None, 1)):
            raise ValueError('view_slice must be a slice with step==1')
        super(ColumnSlicedView, self).__init__(base, index=index, name=name)
        #if the slice is slice(0, None), we have the identity mapping... can we handle that outside?
        self._view_slice = norm_slice(view_slice)

    def _stop(self):
        if self._view_slice.stop is None:
            return len(self._base)
        else:
            return self._view_slice.stop

    @property
    def view_slice(self):
        if self._view_slice.stop is None:
            return slice(self._view_slice.start, self._stop())
        return self._view_slice

    @property
    def shape(self):
        tshape = list(self._base.shape)
        tshape[0] = self._stop() - self._view_slice.start
        return tuple(tshape)

    @property
    def maxshape(self):
        tshape = list(self._base.maxshape)
        tshape[0] = self._stop() - self._view_slice.start
        return tuple(tshape)

    def __len__(self):
        return self.size

    @property
    def size(self):
        return min(self._stop() - self._view_slice.start, self._base.size)

    def resize(self, newsize):
        if newsize==self.size:
            return
        raise ValueError('Cannot resize')

    @property
    def value(self):
        return self._base[self.view_slice]

    def view_to_base(self, index):
        view_slice = self.view_slice
        if isinstance(index, slice):
            index_slice = norm_slice(index)

            start = view_slice.start + index_slice.start
            stop =  view_slice.stop if index_slice.stop is None else view_slice.start+index_slice.stop
            if stop > view_slice.stop:
                logger.error("stop: %s, view_slice.stop: %s", stop, view_slice.stop)
                raise ValueError("Key out of range %s" % index)
            return slice(start, stop, index_slice.step)
        if isinstance(index, Iterable):
            array_index = np.array(index)
            if array_index.dtype == int:
                array_index +=view_slice.start
                if np.any(np.logical_or(array_index < 0, array_index >= view_slice.stop)):
                    raise ValueError("Key out of range %s" % index)
                return array_index
            if array_index.dtype == bool: # not tested yet
                raise NotImplementedError('Indexing with bools not supported yet')
                # I would convert it to indices with np.where and use the fancy indexing code
                mask_start = np.repeat(False,view_slice.start)
                end_size = len(self._base) - len(array_index) -len(mask_start)
                mask_stop =  np.repeat(False, end_size)
                return np.concatenate((mask_start, array_index, mask_stop), axis=0)
            raise ValueError("get_shifted_index not implemented for %s" % index)
        if isinstance(index, integer_types):
            if index < 0:
                index += view_slice.stop
            else:
                index += view_slice.start
            if index<view_slice.start or index>=view_slice.stop:
                raise ValueError("index %d out of range"%index)
            return index
        raise ValueError("view_to_base not implemented for %s" % index)

    def base_to_view(self, index):
        view_slice = self.view_slice
        if isinstance(index, slice):
            index_slice = norm_slice(index)

            start = index_slice.start - view_slice.start
            stop =  None if index_slice.stop is None else index_slice.stop - view_slice.start
            if start <0 or (stop is not None and stop <= start):
                logger.error("stop: %s, view_slice.stop: %s", stop, view_slice.stop)
                raise ValueError("Key out of range %s" % index)
            return slice(start, stop, index_slice.step)
        if isinstance(index, Iterable):
            array_index = np.array(index)
            if array_index.dtype == int:
                array_index -=view_slice.start
                if np.any(array_index[array_index<0]):
                    raise ValueError("Key out of range %s" % index)
                return list(array_index) #[(array_index>=view_slice.start) | (array_index<view_slice.stop)]
            if array_index.dtype == bool: # 
                raise ValueError("not implemented yet")
                # I would convert it to indices with np.where and use the fancy indexing code
                ## mask_start = np.repeat(False,view_slice.start)
                ## end_size = len(self._base) - len(array_index) -len(mask_start)
                ## mask_stop =  np.repeat(False, end_size)
                ## return np.concatenate((mask_start, array_index, mask_stop), axis=0)
            raise ValueError("base_to_view not implemented for %s" % index)
        if isinstance(index, integer_types):
            if index < 0:
                raise ValueError("base_to_view not implemented for %s" % index)
            else:
                index -= view_slice.start
            if index<0:
                raise ValueError("index out of range")
            return index
        raise ValueError("base_to_view not implemented for %s" % index)

    
    def __getitem__(self, index):
        shifted_index = self.view_to_base(index)
        return self._base[shifted_index]

    def __setitem__(self, index, val):
        shifted_index = self.view_to_base(index)
        self._base[shifted_index] = val
        # should be already done by the _base
        #self.update.touch(index)
        #self._base.update.touch(shifted_index)

    def __iter__(self):
        return iter(self.value)


