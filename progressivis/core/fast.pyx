from __future__ import division
import numpy as np
cimport numpy as np

PROP_START_AT_0 = 1
PROP_MONOTONIC_INC = 2
PROP_CONTIGUOUS = 4
PROP_IDENTITY = PROP_CONTIGUOUS | PROP_MONOTONIC_INC | PROP_START_AT_0

def check_contiguity(np.ndarray[np.uint32_t, ndim=1] f, init=0):
    cdef unsigned int len = f.shape[0]
    if len == 0:
        return PROP_IDENTITY
    
    cdef unsigned int prop = PROP_MONOTONIC_INC | PROP_CONTIGUOUS
    cdef unsigned int prev = f[0]
    cdef unsigned int c
    if prev==init:
        prop |= PROP_START_AT_0
    for i in range(1,len):
        c = f[i]
        if c == (prev+1):
            pass # prop &= PROP_START_AT_0 | PROP_MONOTONIC_INC | PROP_CONTIGUOUS
        elif c > prev:
            prop &= ~(PROP_CONTIGUOUS)
        else:
            prop &= ~(PROP_MONOTONIC_INC|PROP_CONTIGUOUS)
            break
        prev = c
            
    return prop

#TODO optimize by splitting according to specific types of indices
def indices_to_slice(indices):
    cdef long int s = 0
    cdef long int e = 0
    for i in indices:
        if e is 0:
            s = i
            e = i+1
        elif i==e:
            e=i+1
        else:
            return indices # not sliceable
    return slice(s, e)

def indices_to_slice_iterator(indices):
    cdef long int s = 0
    cdef long int e = 0
    for i in indices:
        if e is 0:
            s = i
            e = i+1
        elif i==e:
            e=i+1
        else:
            yield slice(s, e)
            s = i
            e = i+1
    if e!=0:
        yield slice(s, e)


# See view-source:http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float
def next_pow2(int v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v+1

