"""
Visualization of a Histogram2D as a heaatmap
"""
from __future__ import annotations

import re
import logging
import base64
import io

import numpy as np
import scipy as sp  # type: ignore
from PIL import Image

from progressivis.core.utils import indices_len
from progressivis.core.slot import SlotDescriptor
from progressivis.table import Table
from progressivis.table.module import TableModule, ReturnRunStep, JSon
from progressivis.stats.histogram2d import Histogram2D

from typing import cast, Optional


logger = logging.getLogger(__name__)


class Heatmap(TableModule):
    "Heatmap module"
    parameters = [
        ("cmax", np.dtype(float), np.nan),
        ("cmin", np.dtype(float), np.nan),
        ("high", np.dtype(int), 65536),
        ("low", np.dtype(int), 0),
        ("filename", np.dtype(object), None),
        ("history", np.dtype(int), 3),
    ]
    inputs = [SlotDescriptor("array", type=Table)]

    # schema = [('image', np.dtype(object), None),
    #           ('filename', np.dtype(object), None),
    #           UPDATE_COLUMN_DESC]
    schema = "{filename: string, time: int64}"

    def __init__(self, colormap: None = None, **kwds):
        super(Heatmap, self).__init__(**kwds)
        self.tags.add(self.TAG_VISUALIZATION)
        self.colormap = colormap
        self.default_step_size = 1
        name = self.generate_table_name("Heatmap")
        self.result = Table(name, dshape=Heatmap.schema, create=True)

    def predict_step_size(self, duration: float) -> int:
        _ = duration
        # Module sample is constant time (supposedly)
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        dfslot = self.get_input_slot("array")
        input_df = dfslot.data()
        dfslot.deleted.next()
        indices = dfslot.created.next()
        steps = indices_len(indices)
        if steps == 0:
            indices = dfslot.updated.next()
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=1)
        histo = input_df.last()["array"]
        if histo is None:
            return self._return_run_step(self.state_blocked, steps_run=1)
        params = self.params
        cmax: Optional[float] = cast(float, params.cmax)
        assert cmax
        if np.isnan(cmax):
            cmax = None
        cmin: Optional[float] = cast(float, params.cmin)
        assert cmin
        if np.isnan(cmin):
            cmin = None
        high: int = cast(int, params.high)
        low: int = cast(int, params.low)
        try:
            if cmin is None:
                cmin = histo.min()
            if cmax is None:
                cmax = histo.max()
            # cscale = cmax - cmin
            scale_hl = float(high - low)
            # scale = float(high - low) / cscale
            # data = (sp.special.cbrt(histo) * 1.0 - cmin) * scale + 0.4999
            data = (sp.special.cbrt(histo) * 1.0 - cmin) * scale_hl + 0.4999
            data[data > high] = high
            data[data < 0] = 0
            data = np.cast[np.uint32](data)
            if low != 0:
                data += low

            image = Image.fromarray(data, mode="I")
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            filename = params.filename
        except Exception:
            image = None
            filename = None
        if filename is not None:
            try:
                if re.search(r"%(0[\d])?d", filename):
                    filename = filename % (run_number)
                filename = self.storage.fullname(self, filename)
                # TODO should do it atomically since it will be
                # called 4 times with the same fn
                image.save(filename, format="PNG")  # bits=16)
                logger.debug("Saved image %s", filename)
                image = None
            except Exception:
                logger.error("Cannot save image %s", filename)
                raise
        else:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", bits=16)
            res = str(base64.b64encode(buffered.getvalue()), "ascii")
            filename = "data:image/png;base64," + res

        table = self.table
        last = table.last()
        if last is None or last["time"] != run_number:
            values = {"filename": filename, "time": run_number}
            table.add(values)
        return self._return_run_step(self.state_blocked, steps_run=1)

    def get_visualization(self) -> str:
        return "heatmap"

    def to_json(self, short=False, with_speed: bool = True) -> JSon:
        json = super(Heatmap, self).to_json(short, with_speed)
        if short:
            return json
        return self.heatmap_to_json(json, short)

    def heatmap_to_json(self, json: JSon, short: bool) -> JSon:
        dfslot = self.get_input_slot("array")
        assert isinstance(dfslot.output_module, Histogram2D)
        histo: Histogram2D = dfslot.output_module
        json["columns"] = [histo.x_column, histo.y_column]
        histo_df = dfslot.data()
        if histo_df is not None and len(histo_df) != 0:
            row = histo_df.last()
            if not (
                np.isnan(row["xmin"])
                or np.isnan(row["xmax"])
                or np.isnan(row["ymin"])
                or np.isnan(row["ymax"])
            ):
                json["bounds"] = {
                    "xmin": row["xmin"],
                    "ymin": row["ymin"],
                    "xmax": row["xmax"],
                    "ymax": row["ymax"],
                }
        df = self.table
        if df is not None and self._last_update != 0:
            row = df.last()
            json["image"] = row["filename"]
        return json

    def get_image(self, run_number: int = None) -> Optional[str]:
        table = self.table
        if table is None or len(table) == 0:
            return None
        last = table.last()
        assert last is not None  # len(table) > 0 so last is not None
        if run_number is None or run_number >= last["time"]:
            run_number = last["time"]
            filename = last["filename"]
        else:
            time = table["time"]
            idx = np.where(time == run_number)[0]
            assert last is not None
            if len(idx) == 0:
                filename = last["filename"]
            else:
                filename = table["filename"][idx[0]]
        return filename

    def get_image_bin(self, run_number: int = None) -> Optional[bytes]:
        file_url = self.get_image(run_number)
        if file_url:
            payload = file_url.split(",", 1)[1]
            return base64.b64decode(payload)
        else:
            return None
