import csv
import inspect
import io
import logging
import pprint

import pandas as pd
import fsspec
from ipywidgets import widgets
# from traitlets import HasTraits, observe, Instance

logger = logging.getLogger(__name__)


def quote_html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


_parser_defaults = {key: val.default
                    for key, val in inspect.signature(pd.read_csv).parameters.items()
                    if val.default is not inspect._empty}

# Borrowed from pandas
MANDATORY_DIALECT_ATTRS = (
    "delimiter",
    "doublequote",
    "escapechar",
    "skipinitialspace",
    "quotechar",
    "quoting",
)


def _merge_with_dialect_properties(dialect, defaults):
    if not dialect:
        return defaults
    kwds = defaults.copy()

    for param in MANDATORY_DIALECT_ATTRS:
        dialect_val = getattr(dialect, param)

        parser_default = _parser_defaults[param]
        provided = kwds.get(param, parser_default)

        # Messages for conflicting values between the dialect
        # instance and the actual parameters provided.
        conflict_msgs = []

        # Don't warn if the default parameter was passed in,
        # even if it conflicts with the dialect (gh-23761).
        if provided != parser_default and provided != dialect_val:
            msg = (
                f"Conflicting values for '{param}': '{provided}' was "
                f"provided, but the dialect specifies '{dialect_val}'. "
                "Using the dialect-specified value."
            )

            # Annoying corner case for not warning about
            # conflicts between dialect and delimiter parameter.
            # Refer to the outer "_read_" function for more info.
            if not (param == "delimiter" and kwds.pop("sep_override", False)):
                conflict_msgs.append(msg)

        if conflict_msgs:
            print("\n\n".join(conflict_msgs))
        kwds[param] = dialect_val
    return kwds


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
        self._df2 = None
        self._rename = None
        self._types = None
        layout = widgets.Layout(border='solid')
        self.head_text = widgets.HTML()
        self.df_text = widgets.HTML()
        self.df2_text = widgets.HTML()
        self.error_msg = widgets.Textarea(description='Error:')
        self.tab = widgets.Tab([self.head_text, self.df_text, self.df2_text])
        for i, title in enumerate(["Head", "DataFrame", "DataFrame2"]):
            self.tab.set_title(i, title)
        self.delimiter = widgets.RadioButtons(
            options=list(zip(self.delimiters, self.del_values)))
        self.delimiter.observe(self._delimiter_cb, names='value')
        self.delim_other = widgets.Text(description='Other:')
        self.delim_other.observe(self._delimiter_cb, names='value')
        self.delimiter = widgets.VBox([
            widgets.Label("Delimiter:"),
            self.delimiter, self.delim_other],
            layout=layout)
        self.columns = widgets.Select(disabled=True,
                                      rows=7)
        self.columns.observe(self._columns_cb, names='value')
        self.column = {}
        self.no_detail = widgets.Label(value="No Column Selected")
        self.details = widgets.Box([
            self.no_detail],
            layout=layout,
            label="Details")
        self.top = widgets.HBox([
            self.delimiter,
            widgets.VBox([
                widgets.Label("Columns:"),
                self.columns,
                ],
                layout=layout),
            self.details])
        self.cmdline = widgets.Textarea(layout=widgets.Layout(width="100%"),
                                        rows=3)
        self.testBtn = widgets.Button(description="Test")
        self.box = widgets.VBox([
            self.top,
            widgets.HBox([self.testBtn,
                          widgets.Label(value="CmdLine:"),
                          self.cmdline]),
            self.tab])
        self.testBtn.on_click(self.test_cmd)
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
        if self._dialect and self._dialect.delimiter == delim:
            return
        self._dialect.delimiter = delim  # TODO check valid delim
        self.delim_other.value = delim
        self.delimiter.value = delim
        self.tab.selected_index = 1
        if self._df is not None:
            self._reset()
        else:
            self.params = _merge_with_dialect_properties(self._dialect,
                                                         self.params)
        self.dataframe(force=True)

    def _reset(self):
        args = self._args.copy()
        self.params = {'index_col': False}
        for name, param in self.signature.parameters.items():
            if name not in ['sep', 'index_col'] and \
               param.default is not inspect._empty:
                self.params[name] = args.pop(name, param.default)
        self.params = _merge_with_dialect_properties(self._dialect,
                                                     self.params)
        self.set_cmdline()
        if args:
            raise ValueError(f"extra keywords arguments {args}")

    def set_cmdline(self):
        params = {}
        for key, val in self.params.items():
            default = _parser_defaults[key]
            if val == default:
                continue
            params[key] = val
        self.cmdline.value = pprint.pformat(params)
        # for key, val in self.params.items():
        #     default = defaults[key].default
        #     if val == default:
        #         continue
        #     if cmdline:
        #         cmdline += ", "
        #     cmdline += f"{key}={repr(val)}"
        # self.cmdline.value = cmdline

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
        # self.params['dialect'] = self._dialect
        self.set_delimiter(self._dialect.delimiter)
        if self.params['header'] == 'infer':
            if sniffer.has_header(head):
                self.params['header'] = 0
        return self._dialect

    def dataframe(self, force=False):
        if not force and self._df is not None:
            return self._df
        self.dialect()
        strin = io.StringIO(self.head())
        try:
            # print(f"read_csv params: {self.params}")
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

    def test_cmd(self, button):
        strin = io.StringIO(self.head())
        try:
            self._df2 = pd.read_csv(strin, **self.params)
        except ValueError as e:
            self._df2 = None
            self.df2_text.value = f'''
<pre style="white-space: pre">Error {quote_html(repr(e))}</pre>
'''
        else:
            with pd.option_context('display.max_rows', self.lines,
                                   'display.max_columns', 0):
                self.df2_text.value = self._df2._repr_html_()
        self.tab.selected_index = 2

    def dataframe_to_params(self):
        df = self._df
        if df is None:
            return
        # if self.params['names'] is None:
        #     self.params['names'] = list(df.columns)
        # # TODO test for existence?
        self.set_cmdline()

    def dataframe_to_columns(self):
        df = self._df
        if df is None:
            self.columns.options = []
            self.columns.disabled = True
            return
        for column in df.columns:
            col = df[column]
            if self.column.get(column) is None:
                col = ColumnInfo(self, col)
                self.column[column] = col
        for column in list(self.column):
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

    def rename_columns(self):
        names = [self.column[col].rename.value
                 for col in self._df.columns]
        self._rename = names
        # print(f"Renames: {names}")

    def usecols_columns(self):
        names = [col for col in self._df.columns
                 if self.column[col].use.value]
        if names == list(self._df.columns):
            del self.params['usecols']
        else:
            self.params['usecols'] = names
        self.set_cmdline()

    def retype_columns(self):
        types = {}
        parse_dates = []
        for name in list(self._df.columns):
            col = self.column[name]
            if col.use.value and col.default_type != col.retype.value:
                type = col.retype.value
                if type == "datetime":
                    types[name] = "str"
                    parse_dates.append(name)
                else:
                    types[name] = type
        if types:
            self._types = types
            self.params['dtype'] = types
        else:
            self._types = None
            del self.params['dtype']
        if parse_dates:
            self.params['parse_dates'] = parse_dates
        else:
            self.params['parse_dates'] = None
        self.set_cmdline()


class ColumnInfo:
    numeric_types = ['int8', 'uint8',
                     'int16' 'uint16',
                     'int32', 'int64',
                     'uint32', 'uint64',
                     'float32', 'float64',
                     'str']
    object_types = ['object', 'str',
                    'category', 'datetime']

    def __init__(self, sniffer, series):
        self.sniffer = sniffer
        self.series = series
        self.default_type = series.dtype.name
        self.name = widgets.Text(description="Column:",
                                 value=series.name,
                                 continuous_update=False,
                                 disabled=True)
        self.type = widgets.Text(description="Type:",
                                 value=series.dtype.name,
                                 disabled=True)
        self.use = widgets.Checkbox(description="Use",
                                    value=True)
        self.use.observe(self.usecols_column, names='value')
        self.rename = widgets.Text(description="Rename:",
                                   value=series.name)
        self.rename.observe(self.rename_column, names='value')
        self.retype = widgets.Dropdown(description="Retype:",
                                       options=self.retype_values(),
                                       value=series.dtype.name)
        self.retype.observe(self.retype_column, names='value')
        self.nunique = widgets.Text(description="Unique vals:",
                                    value=f"{series.nunique()}/{len(series)}")
        self.box = widgets.VBox([self.name, self.rename,
                                 self.type, self.retype,
                                 self.use, self.nunique,
                                 ])

    def retype_values(self):
        type = self.series.dtype.name
        if type in self.numeric_types:
            return self.numeric_types
        elif type == "object":
            return self.object_types
        return type

    def rename_column(self, change):
        self.sniffer.rename_columns()

    def usecols_column(self, change):
        self.sniffer.usecols_columns()

    def retype_column(self, change):
        self.sniffer.retype_columns()
