"""
Visualization of a Histogram2D as a heaatmap
"""

from __future__ import annotations

import re
import logging
import base64
import io
from enum import IntEnum

import numpy as np
import scipy as sp
from scipy.ndimage import gaussian_filter
from PIL import Image

from progressivis.core.module import (
    ReturnRunStep,
    JSon,
    def_input,
    def_output,
    def_parameter,
)
from progressivis.core.api import notNone, indices_len
from progressivis.table.api import PTable
from progressivis.core.module import Module
from progressivis.stats.histogram2d import Histogram2D

from typing import cast, Optional, Any


logger = logging.getLogger(__name__)


class HeatmapTransform(IntEnum):
    NONE = 1
    SQRT = 2
    CBRT = 3
    LOG = 4
    GAUSS = 5


@def_parameter("high", np.dtype(int), 65535)
@def_parameter("low", np.dtype(int), 0)
@def_parameter("filename", np.dtype(object), None)
@def_parameter("history", np.dtype(int), 3)
@def_parameter("transform", np.dtype(int), HeatmapTransform.LOG)
@def_parameter("gaussian_blur", np.dtype(int), 0)
@def_input("array", PTable)
@def_output("result", PTable)
class Heatmap(Module):
    """
    Heatmap module
    """

    # schema = [('image', np.dtype(object), None),
    #           ('filename', np.dtype(object), None),
    #           UPDATE_COLUMN_DESC]
    schema = "{filename: string, time: int64}"

    def __init__(self, colormap: None = None, **kwds: Any) -> None:
        super().__init__(output_required=False, **kwds)
        self.tags.add(self.TAG_VISUALIZATION)
        self.colormap = colormap
        self.default_step_size = 1
        name = self.generate_table_name("Heatmap")
        self.result = PTable(name, dshape=Heatmap.schema, create=True)

    def predict_step_size(self, duration: float) -> int:
        _ = duration
        # Module sample is constant time (supposedly)
        return 1

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        assert self.result is not None
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
        high: int = cast(int, params.high)
        low: int = cast(int, params.low)
        try:
            # Transform
            if params.transform == HeatmapTransform.SQRT:
                data = np.sqrt(histo)
            elif params.transform == HeatmapTransform.CBRT:
                data = sp.special.cbrt(histo)
            elif params.transform == HeatmapTransform.LOG:
                data = np.log1p(histo)
            # elif params.transform == HeatmapTransform.GAUSS:
            #    data = gaussian_filter(histo, sigma=5)
            else:
                data = histo
            if params.gaussian_blur:
                data = gaussian_filter(data, sigma=params.gaussian_blur)
            cmin = data.min()
            cmax = data.max()
            # cscale = cmax - cmin
            # scale_hl = float(high - low)
            scale = float(high - low) / (cmax - cmin)
            # data = (sp.special.cbrt(histo) * 1.0 - cmin) * scale + 0.4999
            data = (data - cmin) * scale + 0.499
            data[data > high] = high
            data[data < 0] = 0
            data = np.asarray(data, dtype=np.uint32)
            if low != 0:
                data += low

            image = Image.fromarray(data, mode="I")
            # image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            filename = params.filename
        except Exception:
            image = None
            filename = None
        if filename is not None and image is not None:
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
            if image is not None:
                image.save(buffered, format="PNG", bits=8)
            res = str(base64.b64encode(buffered.getvalue()), "ascii")
            filename = "data:image/png;base64," + res

        table = self.result
        last = table.last()
        if last is None or last["time"] != run_number:
            values = {"filename": filename, "time": run_number}
            table.add(values)
        return self._return_run_step(self.state_blocked, steps_run=1)

    def get_visualization(self) -> str:
        return "heatmap"

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        json = super().to_json(short, with_speed)
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
        df = self.result
        if df is not None and self._last_update != 0:
            json["image"] = notNone(df.last())["filename"]
        return json

    def get_image(self, run_number: Optional[int] = None) -> Optional[str]:
        filename: Optional[str]
        table = self.result
        if table is None or len(table) == 0:
            return None
        last = notNone(table.last())
        # assert last is not None  # len(table) > 0 so last is not None
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

    def get_image_bin(self, run_number: Optional[int] = None) -> Optional[bytes]:
        file_url = self.get_image(run_number)
        if file_url:
            payload = file_url.split(",", 1)[1]
            return base64.b64decode(payload)
        else:
            return None

    def display_notebook(self, width: int = 512, height: int = 512) -> None:
        import ipywidgets as ipw
        from IPython.display import display

        img = ipw.Image(value=b"\x00", width=width, height=height)
        progress = ipw.IntProgress(
            value=0, min=0, max=1000, description="0/0", orientation="horizontal"
        )
        save = ipw.Button(
            description="Save", disabled=False, button_style="", icon="save"
        )
        box = ipw.VBox([ipw.HBox([progress, save]), img])

        display(box)  # type: ignore

        async def _after_run(m: Module, run_number: int) -> None:
            assert isinstance(m, Heatmap)
            image = m.get_image_bin()  # get the image from the heatmap
            if image is not None:
                img.value = image  # Replace the displayed image with the new one
            prog = m.get_progress()
            if prog is not None:
                value = prog[0]
                max = prog[1]
                progress.value = value
                progress.max = max
                if max != 0:
                    percent = value * 100 / max
                    progress.description = f"{int(percent)}%"

        # TODO: add a on_ending(remove _after_run)
        self.on_after_run(_after_run)  # Install the callback

        def _save(button: ipw.Button) -> None:
            from datetime import datetime

            bytes = self.get_image_bin()
            if bytes is None:
                return
            now = datetime.now()
            fname = f"image-{str(now.replace(microsecond=0))}.png"
            with open(fname, "wb") as fout:
                fout.write(bytes)

        save.on_click(_save)
