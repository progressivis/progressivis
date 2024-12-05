from __future__ import annotations

from ..core.module import (
    Module,
    JSon,
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
    document,
)
from ..core.utils import indices_len, fix_loc
from ..core.slot import Slot
from ..table.api import PTable
from ..utils.psdict import PDict
from fast_histogram import histogram2d  # type: ignore
from ..core.decorators import process_slot, run_if_any

import numpy as np

import logging

from typing import Optional, Tuple, cast, Any, NamedTuple


# Bounds2D = Tuple[float, float, float, float]
class Bounds2D(NamedTuple):
    xmin: float
    xmax: float
    ymin: float
    ymax: float


logger = logging.getLogger(__name__)


@document
@def_parameter(
    "xbins",
    np.dtype(int),
    64,
    doc="the number of ``bins`` (as defined for {{Histogram1D}}) over the ``x`` axis",
)
@def_parameter(
    "ybins",
    np.dtype(int),
    64,
    doc="the number of ``bins`` (as defined for {{Histogram1D}}) over the ``y`` axis",
)
@def_parameter(
    "xdelta",
    np.dtype(float),
    -5,
    doc="the ``delta`` threshold (as defined for {{Histogram1D}}) over the ``x`` axis",
)
@def_parameter(
    "ydelta",
    np.dtype(float),
    -5,
    doc="the ``delta`` threshold (as defined for {{Histogram1D}}) over the ``y`` axis",
)
@def_parameter("history", np.dtype(int), 3)
@def_input("table", PTable, hint_type=dict[str, str], doc="the input table")
@def_input(
    "min",
    PDict,
    doc="The minimum value in the input data. It could be provided by a {{Min}} module",
)
@def_input(
    "max",
    PDict,
    doc="The maximum value in the input data. It could be provided by a {{Max}} module",
)
@def_output(
    "result",
    PTable,
    datashape={"array": np.ndarray,
               "cmin": float,
               "cmax": float,
               "xmin": float,
               "xmax": float,
               "ymin": float,
               "ymax": float,
               "time": int},
    doc="the output table")
class Histogram2D(Module):
    """
    Compute the 2D histogram of two scalar, numerical columns in the input table.
    These two columns are referred to as ``x_column`` and ``y_column`` here.
    """

    schema = (
        "{"
        "array: var * var * float64,"
        "cmin: float64,"
        "cmax: float64,"
        "xmin: float64,"
        "xmax: float64,"
        "ymin: float64,"
        "ymax: float64,"
        "time: int64"
        "}"
    )

    def __init__(
        self,
        x_column: int | str = "",
        y_column: int | str = "",
        with_output: bool = True,
        **kwds: Any,
    ) -> None:
        """
        Args:
            x_column: the name or the position of the ``x`` axis column to be processed
            y_column: the name or the position of the ``y`` axis column to be processed
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(dataframe_slot="table", **kwds)
        self.tags.add(self.TAG_VISUALIZATION)
        self.x_column = x_column
        self.y_column = y_column
        self.default_step_size = 10000
        self.total_read = 0
        self._histo: Optional[np.ndarray[Any, Any]] = None
        self._xedges: Optional[np.ndarray[Any, Any]] = None
        self._yedges: Optional[np.ndarray[Any, Any]] = None
        self._bounds: Optional[Bounds2D] = None
        self._with_output = with_output
        self._heatmap_cache: Optional[Tuple[JSon, Bounds2D]] = None
        self.result = PTable(
            self.generate_table_name("Histogram2D"),
            dshape=Histogram2D.schema,
            chunks={"array": (1, 64, 64)},
            create=True,
        )

    def reset(self) -> None:
        self._histo = None
        self._xedges = None
        self._yedges = None
        self.total_read = 0
        self.get_input_slot("table").reset()
        if self.result:
            self.result.resize(0)

    def is_ready(self) -> bool:
        # If we have created data but no valid min/max, we can only wait
        if self._bounds and self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def get_bounds(self, min_slot: Slot, max_slot: Slot) -> Optional[Bounds2D]:
        min_slot.created.next()
        min_df = min_slot.data()
        assert isinstance(min_df, PDict)
        if min_df is None or len(min_df) == 0:
            return None
        k_ = (lambda x: x) if isinstance(self.x_column, str) else min_df.k_
        xmin: float = cast(float, min_df[k_(self.x_column)])  # type: ignore
        ymin: float = cast(float, min_df[k_(self.y_column)])  # type: ignore
        max_slot.created.next()
        max_df = max_slot.data()
        assert isinstance(max_df, PDict)
        if max_df is None or len(max_df) == 0:
            return None
        k_ = (lambda x: x) if isinstance(self.x_column, str) else max_df.k_
        xmax: float = cast(float, max_df[k_(self.x_column)])  # type: ignore
        ymax: float = cast(float, max_df[k_(self.y_column)])  # type: ignore
        if xmax < xmin:
            xmax, xmin = xmin, xmax
            # logger.warning("xmax < xmin, swapped")
        if ymax < ymin:
            ymax, ymin = ymin, ymax
            # logger.warning("ymax < ymin, swapped")
        return Bounds2D(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    def get_delta(
        self, bounds: Bounds2D  # xmin: float, xmax: float, ymin: float, ymax: float
    ) -> Tuple[float, float]:
        p = self.params
        xdelta, ydelta = p["xdelta"], p["ydelta"]
        if xdelta < 0:
            dx = bounds.xmax - bounds.xmin
            xdelta = dx * xdelta / -100.0
            if np.isnan(xdelta):
                xdelta = 0.
            logger.info("xdelta is %f", xdelta)
        if ydelta < 0:
            dy = bounds.ymax - bounds.ymin
            ydelta = dy * ydelta / -100.0
            if np.isnan(ydelta):
                ydelta = 0.
            logger.info("ydelta is %f", ydelta)
        return (xdelta, ydelta)

    @process_slot("table", reset_if="update", reset_cb="reset")
    @process_slot("min", "max", reset_if=False)
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        assert self.result is not None
        with self.context as ctx:
            dfslot = ctx.table
            min_slot = ctx.min
            max_slot = ctx.max
            if not (
                dfslot.created.any()
                or min_slot.has_buffered()
                or max_slot.has_buffered()
            ):
                logger.info("Input buffers empty")
                return self._return_run_step(self.state_blocked, steps_run=0)
            if self.x_column == "":
                assert self.y_column == ""
                assert dfslot.hint is not None
                assert len(dfslot.hint) == 2
                self.x_column = dfslot.hint["x"]
                self.y_column = dfslot.hint["y"]
            min_slot.clear_buffers()
            max_slot.clear_buffers()
            bounds = self.get_bounds(min_slot, max_slot)
            if bounds is None:
                logger.debug("No bounds yet at run %d", run_number)
                return self._return_run_step(self.state_blocked, steps_run=0)
            b = bounds
            if self._bounds is None:
                (xdelta, ydelta) = self.get_delta(bounds)
                self._bounds = Bounds2D(
                    xmin=b.xmin - xdelta,
                    xmax=b.xmax + xdelta,
                    ymin=b.ymin - ydelta,
                    ymax=b.ymax + ydelta,
                )
                logger.info("New bounds at run %d: %s", run_number, self._bounds)
            else:
                d = self._bounds
                (xdelta, ydelta) = self.get_delta(bounds)
                assert xdelta >= 0 and ydelta >= 0
                # Either the min/max has extended or shrunk beyond the deltas
                if (b.xmin < d.xmin or b.xmax > d.xmax
                    or b.ymin < d.ymin or b.ymax > d.ymax) or (
                    b.xmin > (d.xmin + xdelta)
                    or b.xmax < (d.xmax - xdelta)
                    or b.ymin > (d.ymin + ydelta)
                    or b.ymax < (d.ymax - ydelta)
                ):
                    self._bounds = Bounds2D(
                        xmin=b.xmin - xdelta,
                        xmax=b.xmax + xdelta,
                        ymin=b.ymin - ydelta,
                        ymax=b.ymax + ydelta,
                    )
                    logger.info(
                        "Updated bounds at run %s: %s", run_number, self._bounds
                    )
                    self.reset()
                    dfslot.update(run_number)

            b = self._bounds
            if b.xmin >= b.xmax or b.ymin >= b.ymax:
                logger.error("Invalid bounds: %s", self._bounds)
                return self._return_run_step(self.state_blocked, steps_run=0)

            # Now, we know we have data and bounds, proceed to create a
            # new histogram or to update the previous if is still exists
            # (i.e. no reset)
            p = self.params
            steps = 0
            # if there are new deletions, build the histogram of the del. pairs
            # then subtract it from the main histogram
            if dfslot.base.deleted.any():
                self.reset()
                dfslot.update(run_number)
            elif (
                dfslot.selection.deleted.any() and self._histo is not None
            ):  # i.e. PTableSelectedView
                input_df = dfslot.data().base  # the original table
                # we assume that deletions are only local to the view
                # and the related records still exist in the original table ...
                # TODO : test this hypothesis and reset if false
                raw_indices = dfslot.selection.deleted.next(length=step_size, as_slice=False)
                indices = fix_loc(raw_indices)
                steps += indices_len(indices)
                x = input_df.to_array(locs=indices, columns=[self.x_column]).reshape(-1)
                y = input_df.to_array(locs=indices, columns=[self.y_column]).reshape(-1)
                bins = [p.ybins, p.xbins]
                if len(x) > 0:
                    histo = histogram2d(
                        y, x, bins=bins, range=[[b.ymin, b.ymax], [b.xmin, b.xmax]]
                    )
                    self._histo -= histo
            # if there are new creations, build a partial histogram with them
            # add it to the main histogram
            if not dfslot.created.any():
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            raw_indices = dfslot.created.next(length=step_size, as_slice=False)
            indices = fix_loc(raw_indices)
            steps += indices_len(indices)
            logger.info("Read %d rows", steps)
            self.total_read += steps
            x = input_df.to_array(locs=indices, columns=[self.x_column]).reshape(-1)
            y = input_df.to_array(locs=indices, columns=[self.y_column]).reshape(-1)
            if self._xedges is not None:
                bins = [self._xedges, self._yedges]
            else:
                bins = [p.ybins, p.xbins]
            if len(x) > 0:
                histo = histogram2d(y, x, bins=bins, range=[[b.ymin, b.ymax], [b.xmin, b.xmax]])
            else:
                return self._return_run_step(self.state_blocked, steps_run=0)

            if self._histo is None:
                self._histo = histo
            elif histo is not None:
                self._histo += histo
            if self._histo is not None:
                cmax = self._histo.max()
            values = {
                "array": np.flip(self._histo, axis=0),  # type: ignore
                "cmin": 0,
                "cmax": cmax,
                "xmin": b.xmin,
                "xmax": b.xmax,
                "ymin": b.ymin,
                "ymax": b.ymax,
                "time": run_number,
            }
            if self._with_output:
                table = self.result
                table["array"].set_shape([p.ybins, p.xbins])
                last = table.last()
                if last is None or last["time"] != run_number:
                    table.add(values)
                else:
                    table.loc[last.row] = values
            self.build_heatmap(values)
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)

    def get_visualization(self) -> str:
        return "heatmap"

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        json = super().to_json(short, with_speed)
        if short:
            return json
        return self.heatmap_to_json(json, short)

    def build_heatmap(self, values: JSon) -> None:
        json_: JSon = {}
        row = values
        if not (
            np.isnan(row["xmin"])
            or np.isnan(row["xmax"])
            or np.isnan(row["ymin"])
            or np.isnan(row["ymax"])
        ):
            bounds = Bounds2D(
                xmin=row["xmin"],
                ymin=row["ymin"],
                xmax=row["xmax"],
                ymax=row["ymax"]
            )
            data = row["array"]
            json_["binnedPixels"] = data
            json_["range"] = [np.min(data), np.max(data)]
            json_["count"] = np.sum(data)
            json_["value"] = "heatmap"
            self._heatmap_cache = (json_, bounds)
        return None

    def heatmap_to_json(self, json: JSon, short: bool = False) -> JSon:
        if self._heatmap_cache is None:
            return json
        x_label, y_label = "x", "y"
        domain = ["Heatmap"]
        count = 1
        xmin = ymin = -np.inf
        xmax = ymax = np.inf
        buff, bounds = self._heatmap_cache
        xmin, ymin, xmax, ymax = bounds
        buffers = [buff]
        # TODO: check consistency among classes (e.g. same xbin, ybin etc.)
        xbins, ybins = buffers[0]["binnedPixels"].shape
        encoding = {
            "x": {
                "bin": {"maxbins": xbins},
                "aggregate": "count",
                "field": x_label,
                "type": "quantitative",
                "scale": {"domain": [-7, 7], "range": [0, xbins]},
            },
            "z": {"field": "category", "type": "nominal", "scale": {"domain": domain}},
            "y": {
                "bin": {"maxbins": ybins},
                "aggregate": "count",
                "field": y_label,
                "type": "quantitative",
                "scale": {"domain": [-7, 7], "range": [0, ybins]},
            },
        }
        source = {"program": "progressivis", "type": "python", "rows": count}
        json["chart"] = dict(buffers=buffers, encoding=encoding, source=source)
        json["bounds"] = dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        json["sample"] = dict(data=[], index=[])
        json["columns"] = [x_label, y_label]
        return json
