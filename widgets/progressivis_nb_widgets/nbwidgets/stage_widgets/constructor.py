import ipywidgets as ipw
from progressivis import Scheduler
from progressivis.io import DynVar
from progressivis.core import Sink, aio
from .utils import (make_button, make_loader_box,
                    set_child, dongle_widget, ChainingWidget,
                    get_widget_by_id)

from typing import (
    Any as AnyType,
    Optional,
    Dict,
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
        obj._output_dtypes: Dict[str, AnyType] = {}
        set_child(obj, 1, make_loader_box(obj, ftype="csv"))
        set_child(obj, 2, make_loader_box(obj, ftype="parquet"))
        btn.disabled = True
    return _cbk


class Constructor(ipw.VBox, ChainingWidget):
    last_created = None

    def __init__(self, urls: List[str] = [], *,
                 to_sniff: Optional[str] = None) -> None:
        super().__init__(frame=1,
                         dtypes=None,
                         input_module=None,
                         input_slot=None)
        start_btn = make_button("Start scheduler ...",
                                cb=make_start_scheduler(self))
        self.children = [
            start_btn,
            dongle_widget(""),
            dongle_widget("")
        ]
        print(globals().keys())

    @staticmethod
    def widget_by_id(key):
        return get_widget_by_id(key)
