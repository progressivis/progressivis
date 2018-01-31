from cpython cimport PyObject, Py_INCREF

from khash cimport *
from numpy cimport *

import numpy as np

ONAN = np.nan

cimport cython
cimport numpy as cnp

cnp.import_array()
cnp.import_ufunc()


cdef size_t _INIT_VEC_CAP = 32


cdef class Int64Vector:

    cdef:
        size_t n, m
        ndarray ao
        int64_t *data

    def __cinit__(self):
        self.n = 0
        self.m = _INIT_VEC_CAP
        self.ao = np.empty(_INIT_VEC_CAP, dtype=np.int64)
        self.data = <int64_t*> self.ao.data

    def __len__(self):
        return self.n

    def to_array(self):
        self.ao.resize(self.n)
        self.m = self.n
        return self.ao

    cdef inline append(self, int64_t x):
        if self.n == self.m:
            self.m = max(self.m * 2, _INIT_VEC_CAP)
            self.ao.resize(self.m)
            self.data = <int64_t*> self.ao.data

        self.data[self.n] = x
        self.n += 1

cdef class Int64HashTable:

    def __cinit__(self, size_hint=1):
        self.table = kh_init_int64()
        if size_hint is not None:
            kh_resize_int64(self.table, size_hint)

    def __dealloc__(self):
        kh_destroy_int64(self.table)

    def __contains__(self, object key):
        cdef khiter_t k
        k = kh_get_int64(self.table, key)
        return k != self.table.n_buckets

    def __len__(self):
        return self.table.size

    cpdef get_item(self, int64_t val):
        cdef khiter_t k
        k = kh_get_int64(self.table, val)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(val)

    cpdef del_item(self, int64_t val):
        cdef khiter_t k
        k = kh_get_int64(self.table, val)
        kh_del_int64(self.table, k)

    cpdef set_item(self, int64_t key, Py_ssize_t val):
        cdef:
            khiter_t k
            int ret = 0

        k = kh_put_int64(self.table, key, &ret)
        self.table.keys[k] = key
        if kh_exist_int64(self.table, k):
            self.table.vals[k] = val
        else:
            raise KeyError(key)

    def get_values(self, values):
        if isinstance(values,ndarray):
            return self.get_values_ndarray(values)
        else:
            return self.get_values_generic(values)

    def get_values_ndarray(self, ndarray[int64_t] values):
        cdef:
            Py_ssize_t j = 0, n = self.table.size
            khint_t i
            int64_t key

        if values.size != n:
            raise ValueError('Invalid size %d instead of %d', values.size, n)

        cdef khiter_t k
        for i in range(self.table.n_buckets):
            k = kh_get_int64(self.table, i)
            if k != self.table.n_buckets:
                values[j] = self.table.vals[k]
                j += 1
        return values

    def get_values_generic(self, values):
        cdef:
            Py_ssize_t j = 0, n = self.table.size
            khint_t i
            int64_t key

        if values.size != n:
            raise ValueError('Invalid size %d instead of %d', values.size, n)

        cdef khiter_t k
        for i in range(self.table.n_buckets):
            k = kh_get_int64(self.table, i)
            if k != self.table.n_buckets:
                values[j] = self.table.vals[k]
                j += 1
        return values

    def map(self, keys, ndarray[int64_t] values):
        if isinstance(keys,ndarray):
            self.map_ndarray(keys, values)
        else:
            self.map_generic(keys, values)

    def map_ndarray(self, ndarray[int64_t] keys, ndarray[int64_t] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            int64_t key
            khiter_t k

        for i in range(n):
            key = keys[i]
            k = kh_put_int64(self.table, key, &ret)
            self.table.vals[k] = <Py_ssize_t> values[i]

    def map_generic(self, keys, ndarray[int64_t] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            int64_t key
            khiter_t k

        for i in range(n):
            key = keys[i]
            k = kh_put_int64(self.table, key, &ret)
            self.table.vals[k] = <Py_ssize_t> values[i]

    def get_items(self, ndarray[int64_t] key_value):
        cdef:
            Py_ssize_t i, n = len(key_value)
            int ret = 0
            int64_t key

        for i in range(n):
            key = key_value[i]
            key_value[i] = self.get_item(key)

    def contains_any(self, ndarray[int64_t] key_value):
        cdef:
            Py_ssize_t i, n = len(key_value)
            int64_t key

        for i in range(n):
            key = key_value[i]
            if kh_get_int64(self.table, key) != self.table.n_buckets:
                return True
        return False
