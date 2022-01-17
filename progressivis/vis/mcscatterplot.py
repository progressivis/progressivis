"Multiclass Scatterplot Module."
from __future__ import annotations

from collections import defaultdict

import numpy as np

from progressivis.core.module import Module, ReturnRunStep, JSon
from progressivis.table.nary import NAry
from progressivis.stats import MCHistogram2D, Sample
from progressivis.table.range_query_2d import RangeQuery2d
from progressivis.utils.errors import ProgressiveError
from progressivis.core.utils import is_notebook, get_physical_base
from progressivis.io import DynVar

from typing import (
    Optional,
    Tuple,
    Sequence,
    List,
    Dict,
    cast,
    Union,
    Any,
    Literal,
    TYPE_CHECKING,
)

Bounds = Tuple[float, float, float, float]

if TYPE_CHECKING:
    from progressivis.core.scheduler import Scheduler
    from progressivis.core.slot import Slot


class _DataClass:
    def __init__(
        self,
        name: str,
        that: MCScatterPlot,
        x_column: str,
        y_column: str,
        scheduler: Scheduler,
        approximate: bool = False,
        **kwds: Any,
    ):
        self.name = name
        self._group = that.name
        self.x_column = x_column
        self.y_column = y_column
        self._approximate = approximate
        self._scheduler = scheduler
        self.input_module: Optional[Module] = None
        self.input_slot: Optional[str] = None
        self.min: Any = None
        self.max: Any = None
        self.histogram2d: Optional[MCHistogram2D] = None
        self.heatmap = None
        self.min_value: Optional[DynVar] = that.min_value
        self.max_value: Optional[DynVar] = that.max_value
        self.sample: Union[None, Literal["default"], Module] = None
        self.range_query_2d: Optional[Module] = None

    def scheduler(self) -> Scheduler:
        return self._scheduler

    def create_dependent_modules(
        self,
        input_module: Module,
        input_slot: str,
        histogram2d: Optional[MCHistogram2D] = None,
        heatmap: Optional[Module] = None,
        **kwds: Any,
    ) -> _DataClass:
        if self.input_module is not None:
            return self
        scheduler = self.scheduler()
        with scheduler:
            self.input_module = input_module
            self.input_slot = input_slot
            range_query_2d = RangeQuery2d(
                column_x=self.x_column,
                column_y=self.y_column,
                group=self._group,
                approximate=self._approximate,
                scheduler=scheduler,
            )
            range_query_2d.create_dependent_modules(
                input_module, input_slot, min_value=False, max_value=False
            )
            assert self.min_value is not None and self.max_value is not None
            range_query_2d.input.lower = self.min_value.output.result
            range_query_2d.input.upper = self.max_value.output.result
            if histogram2d is None:
                histogram2d = MCHistogram2D(
                    self.x_column,
                    self.y_column,
                    group=self._group,
                    scheduler=scheduler,
                )
            histogram2d.input.data = range_query_2d.output.result
            if self.sample == "default":
                self.sample = Sample(
                    samples=100, group=self._group, scheduler=scheduler
                )
            if isinstance(self.sample, Sample):
                self.sample.input.table = range_query_2d.output.result
            self.histogram2d = histogram2d
            # self.sample = sample
            # self.select = select
            self.min = range_query_2d.min.output.result
            self.max = range_query_2d.max.output.result
            self.range_query_2d = range_query_2d
            scatterplot = self
            return scatterplot


class MCScatterPlot(NAry):
    "Module visualizing a multiclass scatterplot."

    def __init__(
        self,
        classes: Union[Sequence[Dict[str, Any]], Sequence[Tuple[str, ...]]],
        x_label: str = "x",
        y_label: str = "y",
        approximate: bool = False,
        **kwds: Any,
    ) -> None:
        """Multiclass ..."""
        super(MCScatterPlot, self).__init__(output_required=False, **kwds)
        self.tags.add(self.TAG_VISUALIZATION)
        self._classes = classes  # TODO: check it ...
        self._x_label = x_label
        self._y_label = y_label
        syn_x: Tuple[str, ...]
        syn_y: Tuple[str, ...]
        if isinstance(classes[0], tuple):
            syn_x, syn_y = zip(*[(x, y) for (_, x, y, *ignored) in classes])
        elif isinstance(classes[0], dict):
            syn_x, syn_y = zip(*[(d["x_column"], d["y_column"]) for d in classes])  # type: ignore
        self._translation = {x_label: syn_x, y_label: syn_y}
        self._translated_keys = set(syn_x) | set(syn_y)
        self._approximate = approximate
        self._json_cache: Optional[JSon] = None
        self.input_module: Optional[Module] = None
        self.input_slot: Optional[str] = None
        self._data_class_dict: Dict[str, _DataClass] = {}
        self.min_value: Optional[DynVar] = None
        self.max_value: Optional[DynVar] = None
        self._ipydata: bool = is_notebook()
        self.hist_tensor: Optional[np.ndarray[Any, Any]] = None
        self.sample_tensor: Optional[np.ndarray[Any, Any]] = None

    def forget_changes(self, input_slot: Slot) -> bool:
        changes = False
        if input_slot.deleted.any():
            input_slot.deleted.next()
            changes = True
        if input_slot.created.any():
            input_slot.created.next()
            changes = True
        if input_slot.updated.any():
            input_slot.updated.next()
            changes = True
        return changes

    def get_visualization(self) -> str:
        return "mcscatterplot"

    def predict_step_size(self, duration: float) -> int:
        return 1

    def group_inputs(
        self,
    ) -> Tuple[bool, Dict[str, Dict[str, Tuple[Slot, float, float]]]]:
        """
        Group inputs by classes using meta field on slots
        """
        ret: Dict[str, Dict[str, Tuple[Slot, float, float]]] = defaultdict(dict)
        changes = False
        for name in self.input_slot_names():
            if not self.has_input_slot(name):
                continue
            input_slot = self.get_input_slot(name)
            meta = input_slot.meta
            if meta is None:
                continue
            assert isinstance(meta, dict)
            input_type: str = cast(str, meta["inp"])
            class_: str = cast(str, meta["class_"])
            if input_type not in ("hist", "sample"):
                raise ValueError(f"{input_type} not in [hist, sample]")
            changes |= self.forget_changes(input_slot)
            ret[class_].update(
                {
                    input_type: (
                        input_slot,
                        cast(float, meta["x"]),
                        cast(float, meta["y"]),
                    )
                }
            )
        return changes, ret

    def build_heatmap(self, inp: Slot, domain: Any, plan: int) -> Optional[JSon]:
        inp_table = inp.data()
        if inp_table is None:
            return None
        last = inp_table.last()
        if last is None:
            return None
        row = last.to_dict()
        json_: JSon = {}
        if not (
            np.isnan(row["xmin"])
            or np.isnan(row["xmax"])
            or np.isnan(row["ymin"])
            or np.isnan(row["ymax"])
        ):
            data = row["array"]
            json_["bounds"] = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            if self._ipydata:
                assert isinstance(plan, int)
                json_["binnedPixels"] = plan
                self.hist_tensor[:, :, plan] = row["array"]  # type: ignore
            else:
                data = np.copy(row["array"])  # type: ignore
                json_["binnedPixels"] = data
            json_["range"] = [np.min(data), np.max(data)]  # type: ignore
            json_["count"] = np.sum(data)
            json_["value"] = domain
            return json_
        return None

    def make_json(self, json: JSon) -> JSon:
        buffers = []
        domain = []
        samples: List[Tuple[List[Any], List[Any]]] = []
        count = 0
        xmin = ymin = -np.inf
        xmax = ymax = np.inf
        changes, grouped_inputs = self.group_inputs()
        z = len(grouped_inputs)
        if self._ipydata and self.hist_tensor is None:
            for sl in grouped_inputs.values():
                hi = sl["hist"][0]
                xbins = hi.output_module.params.xbins
                ybins = hi.output_module.params.ybins
                self.hist_tensor = np.zeros((xbins, ybins, z), dtype="int32")
                break
        for i, (cname, inputs) in enumerate(grouped_inputs.items()):
            hist_input = inputs["hist"][0]
            buff = self.build_heatmap(hist_input, cname, i)
            if buff is None:
                return json
            xmin_, ymin_, xmax_, ymax_ = buff.pop("bounds")
            xmin = max(xmin, xmin_)
            ymin = max(ymin, ymin_)
            xmax = min(xmax, xmax_)
            ymax = min(ymax, ymax_)
            buffers.append(buff)
            count += buff["count"]
            domain.append(cname)
            if "sample" in inputs:
                sample_input = inputs["sample"][0]
                select = sample_input.data()
                x_column, y_column = inputs["sample"][1], inputs["sample"][2]
            else:
                select = None

            if self._ipydata:
                smpl: Tuple[List[Any], List[Any]]
                if select is not None:
                    ph_x = get_physical_base(select[x_column])
                    ph_y = get_physical_base(select[y_column])
                    smpl = (
                        ph_x.loc[select[x_column].index.index],
                        ph_y.loc[select[y_column].index.index],
                    )
                else:
                    smpl = ([], [])
            else:
                smpl = (
                    select.to_json(orient="split", columns=[x_column, y_column])
                    if select is not None
                    else []
                )
            samples.append(smpl)
        if self._ipydata:
            samples_counter: List[int] = []
            for vx, vy in samples:
                len_s = len(vx)
                assert len_s == len(vy)
                samples_counter.append(len_s)
            nsam = max(samples_counter)
            self.sample_tensor = np.zeros((nsam, 2, z), dtype="float32")
            for i, (vx, vy) in enumerate(samples):
                if not len(vx):
                    continue
                self.sample_tensor[:, 0, i] = vx
                self.sample_tensor[:, 1, i] = vy
            json["samples_counter"] = samples_counter
            samples = []
        # TODO: check consistency among classes (e.g. same xbin, ybin etc.)
        if self._ipydata:
            assert self.hist_tensor is not None
            xbins, ybins = self.hist_tensor.shape[:-1]
        else:
            xbins, ybins = buffers[0]["binnedPixels"].shape
        encoding = {
            "x": {
                "bin": {"maxbins": xbins},
                "aggregate": "count",
                "field": self._x_label,
                "type": "quantitative",
                "scale": {"domain": [-7, 7], "range": [0, xbins]},
            },
            "z": {"field": "category", "type": "nominal", "scale": {"domain": domain}},
            "y": {
                "bin": {"maxbins": ybins},
                "aggregate": "count",
                "field": self._y_label,
                "type": "quantitative",
                "scale": {"domain": [-7, 7], "range": [0, ybins]},
            },
        }
        source = {"program": "progressivis", "type": "python", "rows": count}
        json["chart"] = dict(buffers=buffers, encoding=encoding, source=source)
        json["bounds"] = dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        s_data: List[float] = []
        # Code note executed and probably wrong
        for i, s in enumerate(samples):
            if not s or not isinstance(s, dict):
                continue
            d = s["data"]
            for row in d:
                row.append(i)
            s_data.extend(d)
        json["sample"] = dict(data=s_data, index=list(range(len(s_data))))
        json["columns"] = [self._x_label, self._y_label]
        if self._ipydata:
            json["hist_tensor"] = self.hist_tensor
            json["sample_tensor"] = self.sample_tensor
        return json

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        for name in self.get_input_slot_multiple(self.nary):
            slot = self.get_input_slot(name)
            # slot.update(run_number)
            if slot.has_buffered():
                slot.clear_buffers()
                self._json_cache = None
        return self._return_run_step(self.state_blocked, steps_run=0)

    def run(self, run_number: int) -> None:
        super(MCScatterPlot, self).run(run_number)
        if self._ipydata:
            return
        if self._json_cache is not None:
            return
        self._json_cache = self._to_json_impl()

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        if self._json_cache:
            return self._json_cache
        self._json_cache = self._to_json_impl(short, with_speed)
        return self._json_cache

    def _to_json_impl(self, short: bool = False, with_speed: bool = True) -> JSon:
        self.image = None
        json = super(MCScatterPlot, self).to_json(short, with_speed=with_speed)
        if short:
            return json
        return self.make_json(json)

    def create_dependent_modules(
        self,
        input_module: Optional[Module] = None,
        input_slot: str = "result",
        sample: str = "default",
        **kwds: Any,
    ) -> None:
        self.input_module = input_module
        self.input_slot = input_slot
        with self.grouped():
            scheduler = self.scheduler()
            self.min_value = DynVar(
                {k: None for k in self._translated_keys},
                translation=self._translation,
                scheduler=scheduler,
            )
            self.max_value = DynVar(
                {k: None for k in self._translated_keys},
                translation=self._translation,
                scheduler=scheduler,
            )
            for cl in self._classes:
                if isinstance(cl, tuple):
                    self._add_class(*cl)   # type: ignore
                elif isinstance(cl, dict):
                    self._add_class(**cl)
                else:
                    raise ValueError(f"Invalid data {cl} in classes")
            self._finalize()

    def __getitem__(self, _class: str) -> _DataClass:
        return self._data_class_dict[_class]

    def _finalize(self) -> None:
        for dc in self._data_class_dict.values():
            assert dc.histogram2d is not None
            for dc2 in self._data_class_dict.values():
                assert dc2.x_column is not None and dc2.y_column is not None
                x, y = dc2.x_column, dc2.y_column
                rq2d = dc2.range_query_2d
                assert rq2d is not None and rq2d.output is not None
                dc.histogram2d.input["table", ("min", x, y)] = rq2d.output.min
                dc.histogram2d.input["table", ("max", x, y)] = rq2d.output.max

    def _add_class(
        self,
        name: str,
        x_column: str,
        y_column: str,
        sample: Union[Literal["default"], Module] = "default",
        sample_slot: str = "result",
        input_module: Optional[Module] = None,
        input_slot: Optional[str] = None,
    ) -> None:
        if self.input_module is None and input_module is None:
            raise ProgressiveError("Input module is not defined!")
        if self.input_module is not None and input_module is not None:
            raise ProgressiveError("Input module is defined twice!")
        if self.input_slot is None and input_slot is None:
            raise ProgressiveError("Input slot is not defined!")
        if (
            self.input_slot is not None
            and input_slot is not None
            and self.input_slot != input_slot
        ):
            raise ProgressiveError("Input slot is defined twice!")
        data_class = _DataClass(
            name,
            self,
            x_column,
            y_column,
            approximate=self._approximate,
            scheduler=self._scheduler,
        )
        data_class.sample = sample
        input_module = input_module or self.input_module
        input_slot = input_slot or self.input_slot
        if input_module is not None and input_slot is not None:
            data_class.create_dependent_modules(input_module, input_slot)
        col_translation = {self._x_label: x_column, self._y_label: y_column}
        hist_meta = dict(inp="hist", class_=name, **col_translation)
        if data_class.histogram2d is not None:
            self.input["table", hist_meta] = data_class.histogram2d.output.result
        if isinstance(data_class.sample, Module):
            meta = dict(inp="sample", class_=name, **col_translation)
            self.input["table", meta] = data_class.sample.output[sample_slot]
        self._data_class_dict[name] = data_class
