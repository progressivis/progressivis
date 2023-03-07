import ipywidgets as ipw
import pandas as pd
from progressivis.io.csv_sniffer import CSVSniffer
from progressivis.io import SimpleCSVLoader
from progressivis.table import PTable
from progressivis.table.constant import Constant
from .utils import (make_button,
                    get_schema, VBoxSchema)
import os

from typing import (
    List, Optional
)


class CsvLoaderW(VBoxSchema):
    def __init__(self) -> None:
        super().__init__()
        self._sniffer: Optional[CSVSniffer] = None
        self._urls: List[str] = []

    def init(self,
             urls: List[str] = [],
             to_sniff: str = "", lines=100) -> None:
        urls_wg = ipw.Textarea(
            value=os.getenv("PROGRESSIVIS_DEFAULT_CSV"),
            placeholder='',
            description='URLS:',
            disabled=False,
            layout=ipw.Layout(width="100%")
        )
        to_sniff_ = ipw.Text(
            value=to_sniff,
            placeholder='',
            description='URL to sniff(optional):',
            disabled=False,
            layout=ipw.Layout(width="100%")
        )
        n_lines = ipw.IntText(
            value=lines,
            description='Rows:',
            disabled=False
        )
        sniff_btn = make_button("Sniff ...",
                                cb=self._sniffer_cb)
        self.schema = dict(
            urls_wg=urls_wg,
            to_sniff=to_sniff_,
            n_lines=n_lines,
            sniff_btn=sniff_btn,
            sniffer=None,
            start_btn=None,
        )

    def _sniffer_cb(self, btn: ipw.Button) -> None:
        urls = self["urls_wg"].value.strip().split("\n")
        assert urls
        self._urls = urls
        to_sniff = self["to_sniff"].value.strip()
        if not to_sniff:
            to_sniff = urls[0]
        n_lines = self["n_lines"].value
        self._sniffer = CSVSniffer(path=to_sniff, lines=n_lines)
        self["sniffer"] = self._sniffer.box
        self["start_btn"] = make_button("Start loading csv ...",
                                        cb=self._start_loader_cb)
        btn.disabled = True
        self["urls_wg"].disabled = True
        self["to_sniff"].disabled = True
        self["n_lines"].disabled = True

    def _start_loader_cb(self, btn: ipw.Button) -> None:
        csv_module = self.init_modules()
        self.output_module = csv_module
        self.output_slot = "result"
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
            filenames = pd.DataFrame({'filename': urls})
            cst = Constant(PTable('filenames', data=filenames), scheduler=s)
            csv = SimpleCSVLoader(scheduler=s, **params)
            csv.input.filenames = cst.output[0]
            sink.input.inp = csv.output.result
            return csv
