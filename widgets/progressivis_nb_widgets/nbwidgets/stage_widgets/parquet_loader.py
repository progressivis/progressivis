import ipywidgets as ipw  # type: ignore
import numpy as np
import pyarrow.parquet as pq
from progressivis.table.dshape import dataframe_dshape  # type: ignore
from progressivis.io import ParquetLoader  # type: ignore
from .utils import (make_button,
                    set_child, dongle_widget, NodeVBox)
import os

from typing import (
    Any as AnyType,
    Optional,
    Dict,
)


def _ds(t):
    ds = dataframe_dshape(t)
    return "datetime64" if ds == "6*uint16" else ds


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


class ParquetLoaderW(NodeVBox):
    last_created = None

    def __init__(self, ctx, url: str = "", dag=None) -> None:
        self._sniffer: Optional[Sniffer] = None
        super().__init__(ctx)
        self._url = ipw.Text(  # type: ignore
            value=os.getenv("PROGRESSIVIS_DEFAULT_PARQUET"),
            placeholder='',
            description='File:',
            disabled=False,
            layout=ipw.Layout(width="100%")
        )
        sniff_btn = make_button("Sniff ...",
                                cb=self._sniffer_cb)
        self._sniffer = None
        self.children = tuple([
            self._url,
            sniff_btn,
            dongle_widget(""),  # sniffer
            dongle_widget(""),  # start loading
            dongle_widget("")   # chaining box
        ])

    def _sniffer_cb(self, btn: ipw.Button) -> None:
        url = self._url.value.strip()
        self._sniffer = Sniffer(url)
        set_child(self, 3, self._sniffer)
        start_btn = make_button("Start loading ...",
                                cb=self._start_loader_cb)
        set_child(self, 4, start_btn)
        btn.disabled = True

    def _start_loader_cb(self, btn: ipw.Button) -> None:
        pq_module = self.init_modules()
        self._output_module = pq_module
        self._output_slot = "result"
        set_child(self, 4, self.make_chaining_box())
        btn.disabled = True
        self.dag.requestAttention(self.title, "widget", "PROGRESS_NOTIFICATION", "")

    def init_modules(self) -> ParquetLoader:
        sink = self._input_module
        s = sink.scheduler()
        with s:
            assert self._sniffer is not None
            cols = list(self._sniffer.get_dtypes().keys())
            pql = ParquetLoader(self._url.value,
                                columns=cols,
                                scheduler=s)
            sink.input.inp = pql.output.result

            def _f(m, rnum):
                if m.table is None:
                    return
                pc = min(100*len(m.table)//self._sniffer.pqfile.metadata.num_rows+1, 100)
                self.dag.updateSummary(self.title, {"progress": pc})
                if pc < 100:
                    self.dag.requestAttention(self.title, "widget",
                                              "PROGRESS_NOTIFICATION", len(m.table))
                else:
                    self.dag.removeRequestAttention(self.title,
                                                    "widget", "PROGRESS_NOTIFICATION")
                    self.dag.requestAttention(self.title, "widget", "STABILITY_REACHED")

            pql.on_after_run(_f)
        return pql

    @property
    def _output_dtypes(self) -> Dict[str, AnyType]:
        assert self._sniffer is not None
        return self._sniffer.get_dtypes()
