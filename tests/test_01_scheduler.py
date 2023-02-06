from __future__ import annotations

from . import ProgressiveTest

from progressivis import Print, Scheduler, ProgressiveError
from progressivis.io import CSVLoader
from progressivis.stats import Min
from progressivis.datasets import get_dataset
from progressivis.core import aio, Sink
from progressivis.core.module import Module, ReturnRunStep, def_input, def_output

from typing import Any


@def_input("a")
@def_input("b", required=False)
@def_output("c")
@def_output("d", required=False)
class TestModule(Module):
    def __init__(self, **kwds: Any) -> None:
        super(TestModule, self).__init__(**kwds)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:  # pragma no cover
        return self._return_run_step(self.state_blocked, 0)


class TestScheduler(ProgressiveTest):
    def test_scheduler(self) -> None:
        with self.assertRaises(ProgressiveError):
            s = Scheduler(0)
        s = Scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"),
            name="csv",
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertIs(s["csv"], csv)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = csv.output.result  # allow csv to start
        check_running = False

        async def _is_running() -> None:
            nonlocal check_running
            check_running = csv.scheduler().is_running()

        aio.run_gather(s.start(), _is_running())

        self.assertTrue(check_running)

        def add_min(s: Scheduler, r: int) -> None:
            with s:
                m = Min(scheduler=s)
                m.input.table = csv.output.result
                prt = Print(proc=self.terse, scheduler=s)
                prt.input.df = m.output.result

        s.on_loop(add_min, 10)
        s.on_loop(self._stop, 20)

        self.assertIs(s["csv"], csv)
        json = s.to_json(short=False)
        self.assertFalse(json["is_running"])
        self.assertTrue(json["is_terminated"])
        html = s._repr_html_()
        self.assertTrue(len(html) != 0)


if __name__ == "__main__":
    ProgressiveTest.main()
