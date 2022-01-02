from __future__ import annotations

import numpy as np
import re
import logging
from io import IOBase

from bz2 import BZ2File
from gzip import GzipFile

from progressivis.table.table import Table
from progressivis.table.module import TableModule, ReturnRunStep

from sklearn.feature_extraction import DictVectorizer  # type: ignore

from typing import Any, Dict, Tuple, List, Callable, Type, Pattern, Match

logger = logging.getLogger(__name__)

PATTERN: Pattern = re.compile(r"\(([0-9]+),([-+.0-9]+)\)[ ]*")


def vec_loader(filename: str, dtype: Type = np.float64) -> Tuple[Any, Dict]:
    """Loads a tf-idf file in .vec format (or .vec.bz2).

    Loads a file and returns a scipy sparse matrix of document features.
    >>> from progressivis.datasets import get_dataset
    >>> mat,features=vec_loader(get_dataset('warlogs'))
    >>> mat.shape
    (3077, 4337)
    """
    openf: Callable[[str], IOBase] = open
    if filename.endswith(".bz2"):
        openf = BZ2File
    elif filename.endswith(".gz") or filename.endswith(".Z"):
        openf = GzipFile
    with openf(filename) as f:
        dataset = []
        for d in f:
            doc = {}
            match: Match
            for match in re.finditer(PATTERN, d.decode("ascii", errors="ignore")):  # type: ignore
                termidx = int(match.group(1))
                termfrx = dtype(match.group(2))
                doc[termidx] = termfrx
            if len(doc) != 0:
                dataset.append(doc)

        dictionary = DictVectorizer(dtype=dtype, sparse=True)
        return (dictionary.fit_transform(dataset), dictionary.get_feature_names())


class VECLoader(TableModule):
    def __init__(self, filename: str, dtype: Type = np.float64, **kwds):
        super(VECLoader, self).__init__(**kwds)
        self._dtype = dtype
        self.default_step_size = kwds.get("chunksize", 10)  # initial guess
        openf: Callable[[str], IOBase] = open
        if filename.endswith(".bz2"):
            openf = BZ2File
        elif filename.endswith(".gz") or filename.endswith(".Z"):
            openf = GzipFile
        self.f: IOBase = openf(filename)
        # When created with a specified chunksize, it returns the parser
        self._rows_read = 0
        self._csr_matrix = None
        self.result = Table(
            self.generate_table_name("documents"),
            dshape="{document: var * float32}",
            fillvalues={"document": 0},
            storagegroup=self.storagegroup,
        )

    def rows_read(self) -> int:
        return self._rows_read

    def toarray(self) -> Any:
        if self._csr_matrix is None:
            docs = self.table["document"]
            dv = DictVectorizer()
            # TODO: race condition when using threads, cleanup_run can reset between
            # setting the value here and returning it at the next instruction
            self._csr_matrix = dv.fit_transform(docs)
        return self._csr_matrix

    def cleanup_run(self, run_number: int) -> int:
        self._csr_matrix = None
        return super(VECLoader, self).cleanup_run(run_number)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if step_size == 0:  # bug
            logger.error("Received a step_size of 0")
            # return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        # print("step_size %d" % step_size)
        if self.f.closed:
            raise StopIteration()

        dataset: List[Dict[int, Any]] = []
        dims = 0
        try:
            while len(dataset) < step_size:
                line = next(self.f)
                line = line.rstrip(b"\n\r")
                if len(line) == 0:
                    continue
                doc: Dict[int, Any] = {}
                for match in re.finditer(PATTERN, line):
                    termidx = int(match.group(1))
                    termfrx: Any = self._dtype(match.group(2))
                    doc[termidx] = termfrx
                    dims = max(dims, termidx)
                if len(doc) != 0:
                    dataset.append(doc)
        except StopIteration:
            self.f.close()

        creates = len(dataset)
        if creates == 0:
            raise StopIteration()

        dims += 1
        documents = self.table["document"]
        if self._rows_read == 0:
            documents.set_shape((dims,))
        else:
            current_dims = documents.shape[1]
            if current_dims < dims:
                documents.set_shape((dims,))
            else:
                dims = current_dims

        self.table.resize(self._rows_read + creates)
        tmp = np.zeros(dims, dtype=np.float64)
        i = self._rows_read
        # with self.lock:
        if True:
            for row in dataset:
                tmp[:] = 0
                for (col, val) in row.items():
                    tmp[col] = val
                documents[i] = tmp
                i += 1
        self._rows_read += creates
        return self._return_run_step(
            self.state_ready, steps_run=creates  # , creates=creates
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
