import ipywidgets as ipw  # type: ignore
from progressivis import Scheduler   # type: ignore
from progressivis.io import DynVar  # type: ignore
from progressivis.core import Sink, aio  # type: ignore
from .utils import (make_button, make_loader_box, get_dag, _Dag,
                    set_child, dongle_widget, ChainingWidget,
                    get_widget_by_id, get_widget_by_key)

from typing import (
    Any as AnyType,
    Optional,
    List,
    Callable,
)


async def _wake_up(sc, sec):
    while True:
        if sc._stopped:
            return
        await aio.sleep(sec)
        await sc.wake_up()


def init_dataflow() -> AnyType:
    s = Scheduler.default = Scheduler()
    with s:
        dyn = DynVar(scheduler=s)
        sink = Sink(scheduler=s)
        sink.input.inp = dyn.output.result
    s.task_start()
    aio.create_task(_wake_up(s, 3))
    return sink


def make_start_scheduler(obj: "Constructor") -> Callable:
    def _cbk(btn: ipw.Button) -> None:
        init_module = init_dataflow()
        obj._output_module = init_module
        obj._output_slot = "result"
        obj._output_dtypes = {}
        set_child(obj, 2, make_loader_box(obj, ftype="csv"))
        set_child(obj, 3, make_loader_box(obj, ftype="parquet"))
        btn.disabled = True
        obj.dag.registerWidget(obj, "root", "root", obj.dom_id, [])
    return _cbk


class Constructor(ipw.VBox, ChainingWidget):
    last_created = None

    def __init__(self, urls: List[str] = [], *, name="root",
                 to_sniff: Optional[str] = None) -> None:
        super().__init__(parent=None,
                         dtypes=None,
                         input_module=None,
                         input_slot=None,
                         dag=_Dag(label=name,
                                  number=0,
                                  dag=get_dag()))
        start_btn = make_button("Start scheduler ...",
                                cb=make_start_scheduler(self))
        self.children = [
            ipw.HTML(f"<h2 id='{self.dom_id}'>{name}</h2>"),
            start_btn,
            dongle_widget(""),
            dongle_widget(""),
            self.dag
        ]

    @staticmethod
    def widget_by_id(key):
        return get_widget_by_id(key)

    @staticmethod
    def widget(key, num=0):
        return get_widget_by_key(key, num)

    @property
    def dom_id(self):
        return f"prog_{id(self)}"

    @property
    def _frame(self):
        return 1
