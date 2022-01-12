from __future__ import annotations

import logging

import numpy as np

from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor
from ..table.table import Table
from ..table.nary import NAry
from ..core.module import Module, ReturnRunStep, JSon
from fast_histogram import histogram2d  # type: ignore

from typing import Optional, Tuple, Any, Dict

Bounds = Tuple[float, float, float, float]

logger = logging.getLogger(__name__)


class MCHistogram2D(NAry):
    parameters = [
        ("xbins", np.dtype(int), 256),
        ("ybins", np.dtype(int), 256),
        ("xdelta", np.dtype(float), -5),  # means 5%
        ("ydelta", np.dtype(float), -5),  # means 5%
        ("history", np.dtype(int), 3),
    ]
    inputs = [SlotDescriptor("data", type=Table, required=True)]

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
        self, x_column: str, y_column: str, with_output: bool = True, **kwds: Any
    ) -> None:
        super(MCHistogram2D, self).__init__(dataframe_slot="data", **kwds)
        self.tags.add(self.TAG_VISUALIZATION)
        self.x_column = x_column
        self.y_column = y_column
        self.default_step_size = 10000
        self.total_read = 0
        self._histo: Optional[np.ndarray[Any, Any]] = None
        self._bounds: Optional[Bounds] = None
        self._with_output = with_output
        self._heatmap_cache: Optional[Tuple[JSon, Bounds]] = None
        self.result = Table(
            self.generate_table_name("MCHistogram2D"),
            dshape=MCHistogram2D.schema,
            chunks={"array": (1, 64, 64)},
            create=True,
        )

    def reset(self) -> None:
        self._histo = None
        self.total_read = 0
        self.get_input_slot("data").reset()
        if self.result:
            self.table.resize(1)

    def predict_step_size(self, duration: float) -> int:
        return Module.predict_step_size(self, duration)

    def is_ready(self) -> bool:
        # If we have created data but no valid min/max, we can only wait
        if self._bounds and self.get_input_slot("data").created.any():
            return True
        return super(MCHistogram2D, self).is_ready()

    def get_delta(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> Tuple[float, float]:
        p = self.params
        xdelta, ydelta = p["xdelta"], p["ydelta"]
        if xdelta < 0:
            dx = xmax - xmin
            xdelta = dx * xdelta / -100.0
            logger.info("xdelta is %f", xdelta)
        if ydelta < 0:
            dy = ymax - ymin
            ydelta = dy * ydelta / -100.0
            logger.info("ydelta is %f", ydelta)
        return (xdelta, ydelta)

    def get_bounds(
        self, run_number: int
    ) -> Optional[Tuple[float, float, float, float, bool]]:
        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        has_creation = False
        min_found = max_found = False
        for name in self.input_slot_names():
            if not self.has_input_slot(name):
                continue
            input_slot = self.get_input_slot(name)
            meta = input_slot.meta
            if meta is None:
                continue
            meta, x_column, y_column = meta
            if meta == "min":
                min_slot = input_slot
                # min_slot.update(run_number)
                if min_slot.has_buffered():
                    has_creation = True
                min_slot.clear_buffers()
                min_df = min_slot.data()
                if min_df is None:
                    continue
                xmin = min(xmin, min_df[x_column])
                ymin = min(ymin, min_df[y_column])
                min_found = True
            elif meta == "max":
                max_slot = input_slot
                # max_slot.update(run_number)
                if max_slot.has_buffered():
                    has_creation = True
                max_slot.clear_buffers()
                max_df = max_slot.data()
                if max_df is None:
                    continue
                xmax = max(xmax, max_df[x_column])
                ymax = max(ymax, max_df[y_column])
                max_found = True
        if not (min_found and max_found):
            return None
        if xmax < xmin:
            xmax, xmin = xmin, xmax
            logger.warning("xmax < xmin, swapped")
        if ymax < ymin:
            ymax, ymin = ymin, ymax
            logger.warning("ymax < ymin, swapped")
        if np.inf in (xmin, -xmax, ymin, -ymax):
            return None
        return (xmin, xmax, ymin, ymax, has_creation)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        dfslot = self.get_input_slot("data")
        # dfslot.update(run_number)
        if dfslot.updated.any():
            logger.debug("reseting histogram")
            self.reset()
            dfslot.update(run_number)
        bounds = self.get_bounds(run_number)
        if bounds is None:
            logger.debug("No bounds yet at run %d", run_number)
            return self._return_run_step(self.state_blocked, steps_run=0)
        xmin, xmax, ymin, ymax, has_creation = bounds
        if not (dfslot.created.any() or has_creation):
            logger.info("Input buffers empty")
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self._bounds is None:
            (xdelta, ydelta) = self.get_delta(xmin, xmax, ymin, ymax)
            self._bounds = (xmin - xdelta, xmax + xdelta, ymin - ydelta, ymax + ydelta)
            logger.info("New bounds at run %d: %s", run_number, self._bounds)
        else:
            (dxmin, dxmax, dymin, dymax) = self._bounds
            (xdelta, ydelta) = self.get_delta(xmin, xmax, ymin, ymax)
            assert xdelta >= 0 and ydelta >= 0

            # Either the min/max has extended, or it has shrunk beyond the deltas
            if (xmin < dxmin or xmax > dxmax or ymin < dymin or ymax > dymax) or (
                xmin > (dxmin + xdelta)
                or xmax < (dxmax - xdelta)
                or ymin > (dymin + ydelta)
                or ymax < (dymax - ydelta)
            ):
                # print('Old bounds: %s,%s,%s,%s'%(dxmin,dxmax,dymin,dymax))
                self._bounds = (
                    xmin - xdelta,
                    xmax + xdelta,
                    ymin - ydelta,
                    ymax + ydelta,
                )
                logger.info("Updated bounds at run %s: %s", run_number, self._bounds)
                self.reset()
                dfslot.update(run_number)

        xmin, xmax, ymin, ymax = self._bounds
        if xmin >= xmax or ymin >= ymax:
            logger.error("Invalid bounds: %s", self._bounds)
            return self._return_run_step(self.state_blocked, steps_run=0)

        # Now, we know we have data and bounds, proceed to create a
        # new histogram or to update the previous if is still exists
        # (i.e. no reset)
        p = self.params
        steps = 0
        # dfslot.data() is a Table() then deleted records are not available
        # anymore => reset
        if dfslot.base.deleted.any():
            self.reset()
            dfslot.update(run_number)
        # else if dfslot.data() is a view and
        # if there are new deletions, build the histogram of the deleted pairs
        # then subtract it from the main histogram
        elif dfslot.selection.deleted.any() and self._histo is not None:
            input_df = dfslot.data().base  # the original table
            # we assume that deletions are only local to the view
            raw_indices = dfslot.deleted.next(length=step_size)
            # and the related records still exist in the original table ...
            # TODO : test this hypothesis and reset if false
            indices = fix_loc(raw_indices)
            steps += indices_len(indices)
            x = input_df.to_array(locs=indices, columns=[self.x_column]).reshape(-1)
            y = input_df.to_array(locs=indices, columns=[self.y_column]).reshape(-1)
            bins = [p.ybins, p.xbins]
            if len(x) > 0:
                histo = histogram2d(y, x, bins=bins, range=[[ymin, ymax], [xmin, xmax]])
                self._histo -= histo
        # if there are new creations, build a partial histogram with them then
        # add it to the main histogram
        if not dfslot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_df = dfslot.data()
        raw_indices = dfslot.created.next(length=step_size)
        indices = fix_loc(raw_indices)
        steps += indices_len(indices)
        logger.info("Read %d rows", steps)
        self.total_read += steps
        x = input_df.to_array(locs=indices, columns=[self.x_column]).reshape(-1)
        y = input_df.to_array(locs=indices, columns=[self.y_column]).reshape(-1)

        bins = [p.ybins, p.xbins]
        if len(x) > 0:
            # using fast_histogram
            histo = histogram2d(y, x, bins=bins, range=[[ymin, ymax], [xmin, xmax]])
        else:
            # histo = None
            # cmax = 0
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
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "time": run_number,
        }
        if self._with_output:
            table = self.table
            table["array"].set_shape([p.ybins, p.xbins])
            # ln = len(table)
            last = table.last()
            # assert last is not None
            if last is None or last["time"] != run_number:
                table.add(values)
            else:
                table.loc[last.row] = values
        self.build_heatmap(values)
        return self._return_run_step(self.next_state(dfslot), steps_run=steps)

    def build_heatmap(self, values: Dict[str, Any]) -> None:
        json_: JSon = {}
        row = values
        if not (
            np.isnan(row["xmin"])
            or np.isnan(row["xmax"])
            or np.isnan(row["ymin"])
            or np.isnan(row["ymax"])
        ):
            bounds = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            data = row["array"]
            # data = sp.special.cbrt(row['array'])
            # json_['data'] = sp.misc.bytescale(data)
            json_["binnedPixels"] = data
            json_["range"] = [np.min(data), np.max(data)]  # type: ignore
            json_["count"] = np.sum(data)
            json_["value"] = "heatmap"
            # return json_
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

    def get_visualization(self) -> str:
        return "heatmap"

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        json = super(MCHistogram2D, self).to_json(short, with_speed=with_speed)
        if short:
            return json
        return self.heatmap_to_json(json, short)
