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
        self.columns = widgets.Select(disabled=True,
                                      rows=7)
        self.columns.observe(self._columns_cb, names='value')
        self.column = {}
        self.no_detail = widgets.Label(value="No Column Selected")
        self.details = widgets.Box([self.no_detail],
                                   label="Details")
        self.top = widgets.HBox([self.delimiter, self.columns, self.details])
        self.box = widgets.VBox([self.top, self.tab])
        self.column_info = []
        self.clear()
        self.dataframe()

    def _delimiter_cb(self, change):
        delim = change['new']
        # print(f"Delimiter: '{delim}'")
        self.set_delimiter(delim)

    def _columns_cb(self, change):
        column = change['new']
        # print(f"Column: '{column}'")
        self.show_column(column)

    def set_delimiter(self, delim):
        if self._dialect.delimiter == delim:
            return
        self.delim_other.value = delim
        self._dialect.delimiter = delim  # TODO check valid delim
        self.delimiter.value = delim
        self.tab.selected_index = 1
        if self._df is not None:
            self._reset()
        self.dataframe(force=True)

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
        try:
            self._df = pd.read_csv(strin, **self.params)
        except ValueError as e:
            self._df = None
            self.df_text.value = f'''
<pre style="white-space: pre">Error {quote_html(repr(e))}</pre>
'''
        else:
            with pd.option_context('display.max_rows', self.lines,
                                   'display.max_columns', 0):
                self.df_text.value = self._df._repr_html_()
        self.dataframe_to_columns()
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

    def dataframe_to_columns(self):
        df = self._df
        if df is None:
            self.columns.options = []
            self.columns.disabled = True
            return
        for column in df.columns:
            col = df[column]
            if self.column.get(column) is None:
                col = ColumnInfo(col)
                self.column[column] = col
        for column in self.column:
            if column not in df.columns:
                col = self.column[column]
                col.box.close()
                del self.column[column]
        self.columns.options = list(df.columns)
        self.columns.disabled = False

    def show_column(self, column):
        if column not in self.column:
            # print(f"Not in columns: '{column}'")
            self.details.children = [self.no_detail]
            return
        col = self.column[column]
        self.details.children = [col.box]


class ColumnInfo:
    numeric_types = ['int32', 'int64',
                     'uint32', 'uint64',
                     'float32', 'float64']
    object_types = ['object', 'string',
                    'categorical', 'datetime']

    def __init__(self, series):
        self.series = series
        self.name = widgets.Text(description="Name:",
                                 value=series.name,
                                 disabled=True)
        self.type = widgets.Text(description="Type:",
                                 value=series.dtype.name,
                                 disabled=True)
        self.use = widgets.Checkbox(description="Use:",
                                    value=True)
        self.nunique = widgets.Text(description="Unique vals:",
                                    value=str(series.nunique()),
                                    disabled=True)
        self.rename = widgets.Text(description="Rename:",
                                   value=series.name)
        self.retype = widgets.Dropdown(description="Retype:",
                                       options=self.retype_values(),
                                       value=series.dtype.name)
        self.box = widgets.VBox([self.name, self.type, self.use, self.nunique,
                                 self.rename, self.retype])

    def retype_values(self):
        type = self.series.dtype.name
        if type in self.numeric_types:
            return self.numeric_types
        elif type == "object":
            return self.object_types
        return type
