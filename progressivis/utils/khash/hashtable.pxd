from khash cimport kh_int64_t, kh_pymap_t, int64_t


cdef class Int64HashTable:
    cdef kh_int64_t *table
    cpdef get_item(self, int64_t val)
    cpdef del_item(self, int64_t val)    
    cpdef set_item(self, int64_t key, Py_ssize_t val)

