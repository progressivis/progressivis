import ipywidgets as ipw  # type: ignore
import pandas as pd
from progressivis.io.csv_sniffer import CSVSniffer  # type: ignore
from progressivis.io import SimpleCSVLoader  # type: ignore
from progressivis.table import Table  # type: ignore
from progressivis.table.module import TableModule  # type: ignore
from progressivis.table.constant import Constant  # type: ignore
from .utils import (make_button, make_chaining_box,
                    set_child, dongle_widget, get_schema, ChainingWidget)
import os

from typing import (
    Any as AnyType,
    Dict,
    List,
    Callable,
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
        csv = SimpleCSVLoader(scheduler=s, throttle=100, **params)
        csv.input.filenames = cst.output[0]
        sink.input.inp = csv.output.result
        return csv


def make_sniffer(obj: "CsvLoaderW") -> Callable:
    def _cbk(btn: ipw.Button) -> None:
        urls = obj._urls_wg.value.strip().split("\n")
        assert urls
        obj._urls = urls
        to_sniff = obj._to_sniff.value.strip()
        if not to_sniff:
            to_sniff = urls[0]
        n_lines = obj._n_lines.value
        obj._sniffer = CSVSniffer(path=to_sniff, lines=n_lines)
        assert obj._sniffer is not None
        set_child(obj, 3, obj._sniffer.box)
        start_btn = make_button("Start loading csv ...",
                                cb=make_start_loader(obj))
        set_child(obj, 4, start_btn)
        btn.disabled = True
    return _cbk


def make_start_loader(obj: "CsvLoaderW") -> Callable:
    def _cbk(btn: ipw.Button) -> None:
        csv_module = init_modules(obj)
        obj._output_module = csv_module
        obj._output_slot = "result"
        set_child(obj, 4, make_chaining_box(obj))
        btn.disabled = True
        obj.dag_running()

    return _cbk


class CsvLoaderW(ipw.VBox, ChainingWidget):
    last_created = None

    def __init__(self,
                 parent: AnyType,
                 dtypes: Dict[str, AnyType],
                 input_module: TableModule,
                 input_slot: str = "result",
                 urls: List[str] = [], *,
                 dag=None,
                 to_sniff: str = "", lines=100) -> None:
        super().__init__(parent=parent,
                         dtypes=dtypes,
                         input_module=input_module,
                         input_slot=input_slot, dag=dag)
        self.dag_register()
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
                                cb=make_sniffer(self))
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

    @property
    def _output_dtypes(self) -> Dict[str, AnyType]:
        return get_schema(self._sniffer)
