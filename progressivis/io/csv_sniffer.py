import csv
import inspect
import io
import logging

import pandas as pd
import fsspec
from ipywidgets import widgets
# from traitlets import HasTraits, observe, Instance

logger = logging.getLogger(__name__)


def quote_html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class CSVSniffer:
    """
    Non progressive class to assist in specifying parameters
    to a CSV module
    """

    signature = inspect.signature(pd.read_csv)
    delimiters = [",", ";", "<TAB>", "<SPACE>", ":"]
    del_values = [",", ";", "\t", " ", ":"]

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
        layout = widgets.Layout(border='solid')
        self.head_text = widgets.HTML()
        self.df_text = widgets.HTML()
        self.tab = widgets.Tab()
        self.tab.children = [self.head_text, self.df_text]
        self.tab.titles = ("Head", "Dataframe")
        self.delimiter = widgets.RadioButtons(
            options=list(zip(self.delimiters, self.del_values)))
        self.delimiter.observe(self._delimiter_cb, names='value')
        self.delim_other = widgets.Text()
        self.delim_other.observe(self._delimiter_cb, names='value')
        self.delimiter = widgets.VBox([self.delimiter, self.delim_other],
                                      description="Delimiter",
                                      layout=layout)
        self.columns = widgets.HBox(label="Columns")
        self.box = widgets.VBox([self.delimiter, self.columns, self.tab])
        self.clear()
        self.dataframe()

    def _delimiter_cb(self, change):
        delim = change['new']
        # print(f"Delimiter: '{delim}'")
        self.set_delimiter(delim)

    def set_delimiter(self, delim):
        if self._dialect.delimiter == delim:
            return
        self.delim_other.value = delim
        self._dialect.delimiter = delim  # TODO check valid delim
        self.delimiter.value = delim
        self.tab.selected_index = 1
        try:
            if self._df is not None:
                self._reset()
            self.dataframe(force=True)
        except ValueError as e:
            self._df = None
            self.df_text.value = f'''
<pre style="white-space: pre">Error {quote_html(repr(e))}</pre>
'''

    def _reset(self):
        args = self._args.copy()
        self.params = {'index_col': False}
        for name, param in self.signature.parameters.items():
            if name != "sep" and param.default is not inspect._empty:
                self.params[name] = args.pop(name, param.default)
        if args:
            raise ValueError(f"extra keywords arguments {args}")

    def clear(self):
        self.lines = 100
        self._head = ''
        self.head_text.value = '<pre style="white-space: pre"></pre>'
        self.df_text.value = ''
        self._dialect = None
        self._reset()

    def _format_head(self):
        self.head_text.value = ('<pre style="white-space: pre">' +
                                quote_html(self._head) +
                                '</pre>')

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
        self._format_head()
        return self._head

    def dialect(self, force=False):
        if not force and self._dialect:
            return self._dialect
        sniffer = csv.Sniffer()
        head = self.head()
        self._dialect = sniffer.sniff(head)
        self.params['dialect'] = self._dialect
        self.set_delimiter(self._dialect.delimiter)
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

    def dataframe(self, force=False):
        if not force and self._df is not None:
            return self._df
        self.params['dialect'] = self.dialect()
        strin = io.StringIO(self.head())
        self._df = pd.read_csv(strin, **self.params)
        self.dataframe_to_params()
        self.df_text.value = self._df._repr_html_()
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
