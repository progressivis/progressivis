import csv
import inspect
import io
import logging

import pandas as pd
import fsspec

logger = logging.getLogger(__name__)


class CSVSniffer:
    """
    Non progressive class to assist in specifying parameters
    to a CSV module
    """

    signature = inspect.signature(pd.read_csv)

    def __init__(self,
                 path,
                 lines=100,
                 **args):
        self.path = path
        self._args = args
        self.lines = 100
        self._head = ""
        self._dialect = None
        self._df = None
        self.clear()

    def clear(self):
        args = self._args.copy()
        self.lines = 100
        self._head = ""
        self._dialect = None
        self.params = {'index_col': False}
        for name, param in self.signature.parameters.items():
            if name != "sep" and param.default is not inspect._empty:
                self.params[name] = args.pop(name, param.default) 
        if args:
            raise ValueError(f"extra keywords arguments {args}")

    def head(self):
        if self._head:
            return self._head
        with fsspec.open(self.path, mode="rt", compression="infer") as inp:
            lineno = 0
            for line in inp:
                if line and lineno < self.lines:
                    self._head += line
                    lineno += 1
                else:
                    break
        return self._head

    def dialect(self):
        if self._dialect:
            return self._dialect
        sniffer = csv.Sniffer()
        head = self.head()
        self._dialect = sniffer.sniff(head)
        self.params['dialect'] = self._dialect
        # self.params['delimiter'] = self._dialect.delimiter
        # self.params['doublequote'] = self._dialect.doublequote
        # self.params['escapechar'] = self._dialect.escapechar
        # self.params['skipinitialspace'] = self._dialect.skipinitialspace
        # self.params['quotechar'] = self._dialect.quotechar
        # self.params['quoting'] = self._dialect.quoting
        if self.params['header'] == 'infer':
            if sniffer.has_header(head):
                self.params['header'] = 0
        return self._dialect

    def dataframe(self):
        if self._df is not None:
            return self._df
        dialect = self.dialect()
        strin = io.StringIO(self.head())
        self._df = pd.read_csv(strin, **self.params)
        self.dataframe_to_params()
        return self._df

    def dataframe_to_params(self):
        df = self._df
        if df is None:
            return
        if self.params['names'] is None:
            self.params['names'] = list(df.columns)
        # TODO test for existence?
        if self.params['usecols'] is None:
            self.params['usecols'] = list(df.columns)
        
