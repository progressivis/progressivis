import ipywidgets as widgets


class ControlPanel(widgets.HBox):
    def __init__(self, scheduler=None):
        self.scheduler = scheduler
        self.bstart = widgets.Button(
            description='Resume',
            disabled=True,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Start',
            icon='play'  # (FontAwesome names without the `fa-` prefix)
        )

        self.bstop = widgets.Button(
            description='Stop',
            disabled=False,
            button_style='',
            tooltip='Stop',
            icon='stop'
        )

        self.bstep = widgets.Button(
            description='Step',
            disabled=True,
            button_style='',
            tooltip='Step',
            icon='step-forward'
        )

        self.run_nb = widgets.HTML(
            value="0",
            placeholder='0',
            description='',
        )
        self.status = "run"
        super().__init__([self.bstart, self.bstop, self.bstep, self.run_nb])

    @property
    def data(self):
        return self.run_nb.value

    @data.setter
    def data(self, val):
        self.run_nb.value = str(val)

    def stop(self):
        self.scheduler.task_stop()
        self.status = "stop"
        self.bstop.disabled = True
        self.bstart.disabled = False
        self.bstep.disabled = False

    def step(self):
        async def _step_once(sched, run_nb):
            await sched.stop()
        self.scheduler.task_start(tick_proc=_step_once)
        self.status = "stop"
        self.bstop.disabled = True
        self.bstart.disabled = False
        self.bstep.disabled = False

    def resume(self):
        self.scheduler.task_start()
        self.status = "run"
        self.bstop.disabled = False
        self.bstart.disabled = True
        self.bstep.disabled = True

    def cb_args(self, key):
        d = dict(resume=(self.bstart, self.resume),
                 stop=(self.bstop, self.stop),
                 step=(self.bstep, self.step))
        return d[key]
