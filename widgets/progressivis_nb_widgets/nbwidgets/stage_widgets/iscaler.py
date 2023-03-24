import time
import ipywidgets as ipw
import weakref
import pandas as pd
from progressivis.core import asynchronize, aio, Sink
from progressivis.io import DynVar
from progressivis.stats.scaling import MinMaxScaler
from typing import Any, Dict, List, Callable, cast
from .utils import make_button, stage_register, VBoxSchema, SchemaBase

spec_no_data = {
    "data": {"name": "data"},
    "height": 500,
    "width": 500,
    "layer": [
        {
            "mark": "bar",
            "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
            "encoding": {
                "x": {
                    "type": "ordinal",
                    "field": "nbins",
                    # "title": "Values", #"axis": {"format": ".2e", "ticks": False},
                    "title": "Values",
                    "axis": {
                        "format": ".2e",
                        "labelExpr": "(datum.value%10>0 ? null : datum.value)",
                    },
                    # "axis": {"labelExpr": "datum.label"},
                },
                "y": {"type": "quantitative", "field": "level", "title": "Count"},
            },
        },
        {
            "mark": "rule",
            "encoding": {
                "x": {
                    "aggregate": "min",
                    "field": "bins",
                    "title": None,
                    "axis": {"tickCount": 0},
                },
                "color": {"value": "red"},
                "size": {"value": 1},
            },
        },
        {
            "mark": "rule",
            "encoding": {
                "x": {
                    "aggregate": "max",
                    "field": "bins",
                    "title": None,
                    "axis": {"tickCount": 0},
                },
                "color": {"value": "red"},
                "size": {"value": 1},
            },
        },
    ],
}


def _refresh_info(wg: Any) -> Callable[..., Any]:
    async def _coro(_1: Any, _2: Any) -> None:
        _ = _1, _2
        await asynchronize(wg.refresh_info)

    return _coro


def refresh_info_hist(hout: Any, hmod: Any) -> None:
    if not hmod.result:
        return
    # spec_with_data = spec_no_data.copy()
    res = hmod.result.last().to_dict()
    hist = res["array"]
    # bins = np.linspace(min_, max_, len(hist))
    source = pd.DataFrame({"nbins": range(len(hist)), "level": hist})
    hout.update("data", remove="true", insert=source)


def _refresh_info_hist(hout: Any, hmod: Any) -> Callable[..., Any]:
    async def _coro(_1: Any, _2: Any) -> None:
        _ = _1, _2
        await asynchronize(refresh_info_hist, hout, hmod)

    return _coro


class IScalerIn(ipw.GridBox):
    def __init__(self, main: "ScalerW") -> None:
        self._main = weakref.ref(main)
        selm = ipw.SelectMultiple(
            options=[(f"{col}:{t}", col) for (col, t) in self.main.dtypes.items()],
            value=[],
            rows=5,
            description="Scaled columns",
            disabled=False,
        )
        selm.observe(self.main._selm_cb, names="value")
        rt = ipw.IntSlider(
            value=1000,
            min=0,
            max=100_000,
            step=1000,
            description="Reset threshold:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        tol = ipw.IntSlider(
            value=5,
            min=0,
            max=100,
            step=1,
            description="Tolerance (%):",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        tol_p100 = ipw.Checkbox(
            value=True, description="Tolerance as %", disabled=False, indent=False
        )
        tol_p100.observe(self._tol_p100_cb, names="value")
        ign = ipw.IntSlider(
            value=10,
            min=0,
            max=1000,
            step=1,
            description="Ignore max:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        btn = ipw.Button(
            description="Apply",
            disabled=True,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Apply",
            icon="check",  # (FontAwesome names without the `fa-` prefix)
        )
        btn.on_click(self._apply_cb)
        self._apply = btn
        rst = ipw.Checkbox(
            value=False, description="Reset:", disabled=False, indent=False
        )
        self._dict = {
            "selm": selm,
            "reset_threshold": rt,
            "delta": tol,
            "delta_p100": tol_p100,
            "ignore_max": ign,
            "reset": rst,
            "apply": btn,
        }
        lst: List[ipw.DOMWidget] = []
        for wg in [selm, rt, tol_p100, tol, ign, rst]:
            lst.append(ipw.Label(wg.description))
            wg.description = ""
            lst.append(wg)
        super().__init__(
            lst + [btn], layout=ipw.Layout(grid_template_columns="repeat(2, 120px)")
        )

    def _tol_p100_cb(self, change: Any) -> None:
        val = change["new"]
        desc = "Tolerance (%):" if val else "Tolerance (abs):"
        for wg in self.children:
            if isinstance(wg, ipw.Label) and wg.value.startswith("Tolerance ("):
                wg.value = desc
                break

    @property
    def delta(self) -> int:
        abs_delta = cast(int, self._dict["delta"].value)
        percent = self._dict["delta_p100"].value
        return -abs_delta if percent else abs_delta

    @property
    def ignore_max(self) -> int:
        return cast(int, self._dict["ignore_max"].value)

    @property
    def selm(self) -> List[str]:
        return cast(List[str], self._dict["selm"].value)

    @property
    def values(self) -> Dict[str, Any]:
        return {k: wg.value for (k, wg) in self._dict.items() if hasattr(wg, "value")}

    def _apply_cb(self, _btn: Any) -> None:
        _ = _btn
        m = self.main.output_module
        values = dict(self.values)  # shallow copy
        values["time"] = time.time()  # always make a change
        # wg._dict['reset'].value = False
        loop = aio.get_running_loop()
        loop.create_task(m.dep.control.from_input(values))

    @property
    def main(self) -> "ScalerW":
        ret = self._main()
        assert ret is not None
        return ret


class IScalerOut(ipw.HBox):
    info_keys = {
        "clipped": "Clipped:",
        "ignored": "Ignored:",
        "needs_changes": "Needs changes:",
        "has_buffered": "Has buff:",
        "last_reset": "Last reset:",
    }

    def __init__(self, main: "ScalerW") -> None:
        self._main = weakref.ref(main)
        self.info_labels: Dict[str, ipw.Label] = {}
        lst = []
        for k, lab in self.info_keys.items():
            lst.append(ipw.Label(lab))
            lst.append(self._info_label(k))
        lst.append(ipw.Label("Rows"))
        self._rows_label = ipw.Label("0")
        lst.append(self._rows_label)
        gb = ipw.GridBox(
            lst, layout=ipw.Layout(grid_template_columns="repeat(2, 120px)")
        )
        super().__init__(children=[gb])
        """
        if not module.hist:
            super().__init__(children=[gb])
        else:
            hlist = []
            for hmod in module.hist.values():
                hout = VegaWidget(spec=spec_no_data)
                hlist.append(hout)
                hmod.on_after_run(_refresh_info_hist(hout, hmod))
            htab = ipw.Tab(hlist)
            for i, t in enumerate(module.hist.keys()):
                htab.set_title(i, t)
            super().__init__(children=[gb, htab])
        """

    def _info_label(self, k: str) -> ipw.Label:
        v = ""
        lab = ipw.Label(v)
        self.info_labels[k] = lab
        return lab

    def refresh_info(self) -> None:
        assert self.main.output_module is not None
        assert hasattr(self.main.output_module, "info")
        assert hasattr(self.main.output_module, "result")
        if not self.main.output_module.info:
            return
        for k, v in self.main.output_module.info.items():
            lab = self.info_labels[k]
            lab.value = str(v)
        if self.main.output_module.result is None:
            return
        self._rows_label.value = str(len(self.main.output_module.result))

    @property
    def main(self) -> "ScalerW":
        ret = self._main()
        assert ret is not None
        return ret


class ScalerW(VBoxSchema):
    class Schema(SchemaBase):
        inp: IScalerIn
        out: IScalerOut
        start_btn: ipw.Button

    child: Schema

    def init(self) -> None:
        self.child.inp = IScalerIn(self)
        self.child.inp.disabled = True
        self.child.out = IScalerOut(self)
        self.child.start_btn = make_button(
            "Run scaler", cb=self._start_btn_cb, disabled=True
        )

    def _start_btn_cb(self, btn: Any) -> None:
        self.output_module = self.init_scaler()
        self.output_slot = "result"
        self.output_module.on_after_run(_refresh_info(self.child.out))
        self.make_chaining_box()
        self.dag_running()
        self.child.inp._dict["selm"].disabled = True
        self.child.inp._dict["apply"].disabled = False

    def _selm_cb(self, change: Any) -> None:
        val = change["new"]
        if val:
            self.child.start_btn.disabled = False
        else:
            self.child.start_btn.disabled = True

    def init_scaler(self) -> MinMaxScaler:
        s = self.input_module.scheduler()
        with s:
            inp = self.child.inp
            dvar = DynVar(
                {"delta": inp.delta, "ignore_max": inp.ignore_max}, scheduler=s
            )
            sc = MinMaxScaler(reset_threshold=10_000, usecols=inp.selm)
            sc.create_dependent_modules(self.input_module, hist=True)
            sc.dep.control = dvar
            sc.input.control = dvar.output.result
            sink = Sink(scheduler=s)
            sink.input.inp = sc.output.info
            sink2 = Sink(scheduler=s)
            sink2.input.inp = sc.output.result
        return sc


stage_register["Scaler"] = ScalerW
