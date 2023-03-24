import ipywidgets as ipw
from progressivis import Scheduler
from progressivis.io import DynVar
from progressivis.core import Sink, aio
from .utils import (
    make_button,
    get_dag,
    _Dag,
    DAGWidget,
    RootVBox,
    SchemaBox,
    NodeVBox,
    SchemaBase,
    get_widget_by_id,
    get_widget_by_key,
)

from typing import (
    Any as AnyType,
    Optional,
    List,
)


async def _wake_up(sc: Scheduler, sec: float) -> None:
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


class Constructor(RootVBox, SchemaBox):
    class Schema(SchemaBase):
        h2: ipw.HTML
        start_btn: ipw.Button
        csv: Optional[ipw.HBox]
        parquet: Optional[ipw.HBox]
        dag: DAGWidget

    child: Schema

    last_created = None

    def __init__(
        self,
        urls: List[str] = [],
        *,
        name: str = "root",
        to_sniff: Optional[str] = None,
    ) -> None:
        ctx = dict(
            parent=None,
            dtypes=None,
            input_module=None,
            input_slot=None,
            dag=_Dag(label=name, number=0, dag=get_dag()),
        )
        RootVBox.__init__(self, ctx)
        SchemaBox.__init__(self)
        self.child.start_btn = make_button(
            "Start scheduler ...", cb=self._start_scheduler_cb
        )
        self.child.h2 = ipw.HTML(f"<h2 id='{self.dom_id}'>{name}</h2>")
        self.child.dag = self.dag

    def _start_scheduler_cb(self, btn: ipw.Button) -> None:
        init_module = init_dataflow()
        self._output_module = init_module
        self._output_slot = "result"
        self._output_dtypes = {}
        self.child.csv = self.make_loader_box(ftype="csv")
        self.child.parquet = self.make_loader_box(ftype="parquet")
        btn.disabled = True
        self.dag.registerWidget(self, "root", "root", self.dom_id, [])

    @staticmethod
    def widget_by_id(key: int) -> NodeVBox:
        return get_widget_by_id(key)

    @staticmethod
    def widget(key: str, num: int = 0) -> NodeVBox:
        return get_widget_by_key(key, num)

    @property
    def dom_id(self) -> str:
        return f"prog_{id(self)}"

    @property
    def _frame(self) -> int:
        return 1

    def dag_register(self) -> None:
        pass
