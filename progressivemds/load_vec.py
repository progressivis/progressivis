# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix

import re
from bz2 import BZ2File

def load_vec(filename, dtype=np.float64):
    pattern = re.compile(r"\(([0-9]+),([-+.0-9]+)\)[ ]*")
    openf=open
    if filename.endswith('.bz2'):
        openf=BZ2File
    with openf(filename) as f:
        indptr = [0]
        indices = []
        data = []
        for d in f.readlines():
            for match in re.finditer(pattern, d):
                termidx = int(match.group(1))
                termfrx = dtype(match.group(2))
                indices.append(termidx)
                data.append(termfrx)
            indptr.append(len(indices))

        return csr_matrix((data, indices, indptr), dtype=dtype)

