import ipywidgets as ipw
import numpy as np
import pyarrow.parquet as pq
from progressivis.table.dshape import dataframe_dshape as _ds
from progressivis.io import ParquetLoader
from progressivis.table.module import TableModule
from .utils import (make_button, make_chaining_box,
                    set_child, dongle_widget, ChainingWidget)
import os

from typing import (
    Any as AnyType,
    Dict,
    Callable,
)


def init_modules(obj: "ParquetLoaderW") -> ParquetLoader:
    sink = obj._input_module
    s = sink.scheduler()
    with s:
        cols = list(obj._sniffer.get_dtypes().keys())
        pq = ParquetLoader(obj._url.value,
                           columns=cols,
                           scheduler=s)
        sink.input.inp = pq.output.result
    return pq


class ColInfo(ipw.VBox):
    def __init__(self, raw_info, dtype, *args, **kw):
        super().__init__(*args, **kw)
        self.info = ipw.Textarea("\n".join(str(raw_info).strip().split("\n")[1:]),
                                 rows=12)
        self.use = ipw.Checkbox(description="Use", value=True)
        self.dtype = dtype
        self.children = [self.info, self.use]


class Sniffer(ipw.HBox):
    def __init__(self, url, *args, **kw):
        super().__init__(*args, **kw)
        self.pqfile = pq.ParquetFile(url)
        self.schema = self.pqfile.schema.to_arrow_schema()
        names = self.schema.names
        self.names = names
        types = [t.to_pandas_dtype() for t in self.schema.types]
        decorated = [(f"{n}:{np.dtype(t).name}", n) for (n, t) in zip(names, types)]
        self.info_cols = {n: ColInfo(self.pqfile.schema.column(i), np.dtype(types[i]))
                          for (i, n) in enumerate(names)}
        # Column selection
        self.columns = ipw.Select(disabled=False, rows=7, options=decorated)
        self.columns.observe(self._columns_cb, names="value")
        # Column details
        self.column: Dict[str, ColInfo] = {}
        self.no_detail = ipw.Label(value="No Column Selected")
        self.details = ipw.Box([self.no_detail], label="Details")
        layout = ipw.Layout(border="solid")
        # Toplevel Box
        self.children = (
                ipw.VBox([ipw.Label("Columns"), self.columns], layout=layout),
                ipw.VBox(
                    [ipw.Label("Selected Column"), self.details], layout=layout
                ),
        )

    def get_dtypes(self):
        return {k: _ds(col.dtype)
                for (k, col) in self.info_cols.items()
                if col.use.value}

    def _columns_cb(self, change: Dict[str, AnyType]) -> None:
        column = change["new"]
        self.show_column(column)

    def show_column(self, column: str) -> None:
        if column not in self.names:
            self.details.children = [self.no_detail]
            return
        col = self.info_cols[column]
        self.details.children = [col]


def make_sniffer(obj: "ParquetLoaderW") -> Callable:
    def _cbk(btn: ipw.Button) -> None:
        url = obj._url.value.strip()
        obj._sniffer = Sniffer(url)
        set_child(obj, 3, obj._sniffer)
        start_btn = make_button("Start loading ...",
                                cb=make_start_loader(obj))
        set_child(obj, 4, start_btn)
        btn.disabled = True
    return _cbk


def make_start_loader(obj: "ParquetLoaderW") -> Callable:
    def _cbk(btn: ipw.Button) -> None:
        pq_module = init_modules(obj)
        obj._output_module = pq_module
        obj._output_slot = "result"
        set_child(obj, 4, make_chaining_box(obj))
        btn.disabled = True
    return _cbk


class ParquetLoaderW(ipw.VBox, ChainingWidget):
    last_created = None

    def __init__(self,
                 frame: AnyType,
                 dtypes: Dict[str, AnyType],
                 input_module: TableModule,
                 input_slot: str = "result",
                 url: str = "") -> None:
        super().__init__(frame=frame,
                         dtypes=dtypes,
                         input_module=input_module,
                         input_slot=input_slot)
        self._url = ipw.Text(
            value=os.getenv("PROGRESSIVIS_DEFAULT_PARQUET"),
            placeholder='',
            description='File:',
            disabled=False,
            layout=ipw.Layout(width="100%")
        )
        sniff_btn = make_button("Sniff ...",
                                cb=make_sniffer(self))
        self._sniffer = None
        self.children = [
            self._url,
            sniff_btn,
            dongle_widget(""),  # sniffer
            dongle_widget(""),  # start loading
            dongle_widget("")   # chaining box
        ]

    @property
    def _output_dtypes(self) -> Dict[str, AnyType]:
        return self._sniffer.get_dtypes()
