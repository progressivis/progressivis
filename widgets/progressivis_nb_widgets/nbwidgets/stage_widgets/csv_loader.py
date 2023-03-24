import ipywidgets as ipw
import pandas as pd
from progressivis.io.csv_sniffer import CSVSniffer
from progressivis.io import SimpleCSVLoader
from progressivis.table import PTable
from progressivis.table.constant import Constant
from .utils import make_button, get_schema, VBoxSchema, SchemaBase
import os

from typing import List, Optional


class CsvLoaderW(VBoxSchema):
    class Schema(SchemaBase):
        urls_wg: ipw.Textarea
        to_sniff: ipw.Text
        n_lines: ipw.IntText
        sniff_btn: ipw.Button
        sniffer: Optional[CSVSniffer]
        start_btn: Optional[ipw.Button]

    child: Schema

    def __init__(self) -> None:
        super().__init__()
        self._sniffer: Optional[CSVSniffer] = None
        self._urls: List[str] = []

    def init(self, urls: List[str] = [], to_sniff: str = "", lines: int = 100) -> None:
        self.child.urls_wg = ipw.Textarea(
            value=os.getenv("PROGRESSIVIS_DEFAULT_CSV"),
            placeholder="",
            description="URLS:",
            disabled=False,
            layout=ipw.Layout(width="100%"),
        )
        self.child.to_sniff = ipw.Text(
            value=to_sniff,
            placeholder="",
            description="URL to sniff(optional):",
            disabled=False,
            layout=ipw.Layout(width="100%"),
        )
        self.child.n_lines = ipw.IntText(
            value=lines, description="Rows:", disabled=False
        )
        self.child.sniff_btn = make_button("Sniff ...", cb=self._sniffer_cb)

    def _sniffer_cb(self, btn: ipw.Button) -> None:
        urls = self.child.urls_wg.value.strip().split("\n")
        assert urls
        self._urls = urls
        to_sniff = self.child.to_sniff.value.strip()
        if not to_sniff:
            to_sniff = urls[0]
        n_lines = self.child.n_lines.value
        self._sniffer = CSVSniffer(path=to_sniff, lines=n_lines)
        self.child.sniffer = self._sniffer.box
        self.child.start_btn = make_button(
            "Start loading csv ...", cb=self._start_loader_cb
        )
        btn.disabled = True
        self.child.urls_wg.disabled = True
        self.child.to_sniff.disabled = True
        self.child.n_lines.disabled = True

    def _start_loader_cb(self, btn: ipw.Button) -> None:
        csv_module = self.init_modules()
        self.output_module = csv_module
        self.output_slot = "result"
        assert self._sniffer is not None
        self.output_dtypes = get_schema(self._sniffer)
        self.make_chaining_box()
        btn.disabled = True
        self.dag_running()

    def init_modules(self) -> SimpleCSVLoader:
        urls = self._urls
        assert self._sniffer is not None
        params = self._sniffer.params
        sink = self.input_module
        s = sink.scheduler()
        with s:
            filenames = pd.DataFrame({"filename": urls})
            cst = Constant(PTable("filenames", data=filenames), scheduler=s)
            csv = SimpleCSVLoader(scheduler=s, **params)
            csv.input.filenames = cst.output[0]
            sink.input.inp = csv.output.result
            return csv
