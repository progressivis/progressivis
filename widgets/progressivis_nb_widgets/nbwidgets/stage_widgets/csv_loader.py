import ipywidgets as ipw  # type: ignore
import pandas as pd
from progressivis.io.csv_sniffer import CSVSniffer  # type: ignore
from progressivis.io import SimpleCSVLoader  # type: ignore
from progressivis.table import Table  # type: ignore
from progressivis.table.constant import Constant  # type: ignore
from .utils import (make_button,
                    set_child, dongle_widget, get_schema, NodeVBox)
import os

from typing import (
    Any as AnyType,
    Dict,
    List,
)


def init_modules(obj: "CsvLoaderW") -> SimpleCSVLoader:
    urls = obj._urls
    assert obj._sniffer is not None
    params = obj._sniffer.params
    sink = obj._input_module
    s = sink.scheduler()
    with s:
        filenames = pd.DataFrame({'filename': urls})
        cst = Constant(Table('filenames', data=filenames), scheduler=s)
        csv = SimpleCSVLoader(scheduler=s, **params)
        csv.input.filenames = cst.output[0]
        sink.input.inp = csv.output.result
        return csv


class CsvLoaderW(NodeVBox):
    last_created = None

    def __init__(self, ctx,
                 urls: List[str] = [],
                 to_sniff: str = "", lines=100) -> None:
        super().__init__(ctx)
        self._urls_wg = ipw.Textarea(
            value=os.getenv("PROGRESSIVIS_DEFAULT_CSV"),
            placeholder='',
            description='URLS:',
            disabled=False,
            layout=ipw.Layout(width="100%")
        )
        self._urls = None
        self._to_sniff = ipw.Text(
            value=to_sniff,
            placeholder='',
            description='URL to sniff(optional):',
            disabled=False,
            layout=ipw.Layout(width="100%")
        )
        self._n_lines = ipw.IntText(
            value=lines,
            description='Lines:',
            disabled=False
        )
        sniff_btn = make_button("Sniff ...",
                                cb=self._sniffer_cb)
        self._sniffer = None
        self.children = [
            self._urls_wg,
            self._to_sniff,
            self._n_lines,
            sniff_btn,
            dongle_widget(""),
            dongle_widget(""),
            dongle_widget("")
        ]

    def _sniffer_cb(self, btn: ipw.Button) -> None:
        urls = self._urls_wg.value.strip().split("\n")
        assert urls
        self._urls = urls
        to_sniff = self._to_sniff.value.strip()
        if not to_sniff:
            to_sniff = urls[0]
        n_lines = self._n_lines.value
        self._sniffer = CSVSniffer(path=to_sniff, lines=n_lines)
        assert self._sniffer is not None
        set_child(self, 3, self._sniffer.box)
        start_btn = make_button("Start loading csv ...",
                                cb=self._start_loader_cb)
        set_child(self, 4, start_btn)
        btn.disabled = True

    def _start_loader_cb(self, btn: ipw.Button) -> None:
        csv_module = init_modules(self)
        self._output_module = csv_module
        self._output_slot = "result"
        set_child(self, 4, self.make_chaining_box())
        btn.disabled = True
        self.dag_running()

    @property
    def _output_dtypes(self) -> Dict[str, AnyType]:
        return get_schema(self._sniffer)
