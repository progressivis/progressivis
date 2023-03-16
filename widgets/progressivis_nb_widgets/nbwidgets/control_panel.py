from __future__ import annotations

import ipywidgets as widgets

from typing import Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from progressivis.core.scheduler import Scheduler


class ControlPanel(widgets.HBox):  # pylint: disable=too-many-ancestors
    def __init__(self, scheduler: Optional[Scheduler]) -> None:
        self.scheduler = scheduler
        self.bstart = widgets.Button(
            description="Resume",
            disabled=True,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Start",
            icon="play",  # (FontAwesome names without the `fa-` prefix)
        )

        self.bstop = widgets.Button(
            description="Stop",
            disabled=False,
            button_style="",
            tooltip="Stop",
            icon="stop",
        )

        self.bstep = widgets.Button(
            description="Step",
            disabled=True,
            button_style="",
            tooltip="Step",
            icon="step-forward",
        )

        self.run_nb = widgets.HTML(value="0", placeholder="0", description="",)
        self.status = "run"
        super().__init__([self.bstart, self.bstop, self.bstep, self.run_nb])
        self.bstart.on_click(self.resume)
        self.bstop.on_click(self.stop)
        self.bstep.on_click(self.step)

    @property
    def data(self) -> Any:
        return self.run_nb.value

    @data.setter
    def data(self, val: Any) -> None:
        self.run_nb.value = str(val)

    def stop(self, _btn: Any) -> None:
        _ = _btn
        assert self.scheduler is not None
        self.scheduler.task_stop()
        self.status = "stop"
        self.bstop.disabled = True
        self.bstart.disabled = False
        self.bstep.disabled = False

    def step(self, _btn: Any) -> None:
        _ = _btn

        async def _step_once(sched: Scheduler, _: Any) -> None:
            await sched.stop()
        assert self.scheduler is not None
        self.scheduler.task_start(tick_proc=_step_once)
        self.status = "stop"
        self.bstop.disabled = True
        self.bstart.disabled = False
        self.bstep.disabled = False

    def resume(self, _btn: Any) -> None:
        _ = _btn
        assert self.scheduler is not None
        self.scheduler.task_start()
        self.status = "run"
        self.bstop.disabled = False
        self.bstart.disabled = True
        self.bstep.disabled = True
