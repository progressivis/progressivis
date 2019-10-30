from __future__ import absolute_import, division, print_function

import numpy as np
import six
import re
import logging
logger = logging.getLogger(__name__)

from bz2 import BZ2File
from gzip import GzipFile

from progressivis.table.table import Table
from progressivis.table.module import TableModule

from sklearn.feature_extraction import DictVectorizer

PATTERN = re.compile(r'\(([0-9]+),([-+.0-9]+)\)[ ]*')


def vec_loader(filename, dtype=np.float64):
    """Loads a tf-idf file in .vec format (or .vec.bz2).

    Loads a file and returns a scipy sparse matrix of document features.
    >>> from progressivis.datasets import get_dataset
    >>> mat,features=vec_loader(get_dataset('warlogs'))
    >>> mat.shape
    (3077, 4337)
    """
    openf=open
    if filename.endswith('.bz2'):
        openf=BZ2File
    elif filename.endswith('.gz') or filename.endswith('.Z'):
        openf=GzipFile
    with openf(filename) as f:
        dataset = []
        for d in f:
            doc = {}
            for match in re.finditer(PATTERN, d.decode('ascii', errors='ignore')):
                termidx = int(match.group(1))
                termfrx = dtype(match.group(2))
                doc[termidx] = termfrx
            if len(doc)!=0:
                dataset.append(doc) 

        dictionary = DictVectorizer(dtype=dtype,sparse=True)
        return (dictionary.fit_transform(dataset), dictionary.get_feature_names())


class VECLoader(TableModule):
    def __init__(self, filename, dtype=np.float64, **kwds):
        super(VECLoader, self).__init__(**kwds)
        self._dtype = dtype
        self.default_step_size = kwds.get('chunksize', 10)  # initial guess
        openf=open
        if filename.endswith('.bz2'):
            openf=BZ2File
        elif filename.endswith('.gz') or filename.endswith('.Z'):
            openf=GzipFile
        self.f = openf(filename)
        # When created with a specified chunksize, it returns the parser
        self._rows_read = 0
        self._csr_matrix = None
        self._table = Table(self.generate_table_name('documents'),
                            dshape="{document: var * float32}",
                            fillvalues={'document': 0},
                            storagegroup=self.storagegroup)

    def rows_read(self):
        return self._rows_read

    def toarray(self):
        if self._csr_matrix is None:
            docs = self.table()['document']
            dv=DictVectorizer()
            #TODO: race condition when using threads, cleanup_run can reset between
            #setting the value here and returning it at the next instruction
            self._csr_matrix = dv.fit_transform(docs)
        return self._csr_matrix

    def cleanup_run(self, run_number):
        self._csr_matrix = None
        super(VECLoader, self).cleanup_run(run_number)

    def run_step(self,run_number,step_size, howlong):
        if step_size==0: # bug
            logger.error('Received a step_size of 0')
            #return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        print('step_size %d'%step_size)
        if self.f is None:
            raise StopIteration()

        dataset = []
        dims = 0
        try:
            while len(dataset) < step_size:
                line = next(self.f)
                line=line.rstrip(b'\n\r')
                if len(line)==0:
                    continue
                doc = {}
                for match in re.finditer(PATTERN, line):
                    termidx = int(match.group(1))
                    termfrx = self._dtype(match.group(2))
                    doc[termidx] = termfrx
                    dims = max(dims, termidx)
                if len(doc)!=0:
                    dataset.append(doc)
        except StopIteration:
            self.f.close()
            self.f = None

        creates = len(dataset)
        if creates==0:
            raise StopIteration()

        dims += 1
        documents = self._table['document']
        if self._rows_read == 0:
            documents.set_shape((dims,))
        else:
            current_dims = documents.shape[1]
            if current_dims < dims:
                documents.set_shape((dims,))
            else:
                dims = current_dims

        self._table.resize(self._rows_read+creates)
        tmp = np.zeros(dims, dtype=np.float)
        i = self._rows_read
        #with self.lock:
        if True:
            for row in dataset:
                tmp[:] = 0
                for (col, val) in six.iteritems(row):
                    tmp[col] = val
                documents[i] = tmp
                i += 1
        self._rows_read += creates
        return self._return_run_step(self.state_ready, steps_run=creates, creates=creates)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
