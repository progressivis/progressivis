from __future__ import annotations

from progressivis.core import JSONEncoderNp as JSON
from progressivis.table.paging_helper import PagingHelper
from .utils import update_widget
from .data_table import DataPTable

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis import Module


debug_console = None
_dmp = JSON.dumps
# https://datatables.net/examples/basic_init/alt_pagination.html


class SlotWg(DataPTable):
    def __init__(
        self, module: Module, slot_name: str, dconsole: Optional[Any] = None
    ) -> None:
        global debug_console  # pylint: disable=global-statement
        debug_console = dconsole
        self.module = module
        self.slot_name = slot_name
        super().__init__()

    async def refresh(self) -> None:
        tbl = self.module.get_data(self.slot_name)
        if tbl is None:
            return
        if not self.columns:
            await update_widget(
                self, "dt_id", "dt_" + self.module.name + "_" + self.slot_name
            )
            if isinstance(tbl, dict):
                await update_widget(self, "columns", _dmp(["index"] + list(tbl.keys())))
            else:
                await update_widget(self, "columns", _dmp(["index"] + tbl.columns))
        if isinstance(tbl, dict):
            await update_widget(
                self,
                "data",
                _dmp(
                    {
                        "draw": 1,
                        "recordsTotal": 1,
                        "recordsFiltered": 1,
                        "data": [[0] + list(tbl.values())],
                    }
                ),
            )
        else:
            _len = len(tbl)
            size = 10
            info = dict(start=0, end=size, draw=1)
            if self.page:
                info.update(self.page)
            start = info["start"]
            end = info["end"]
            draw = info["draw"]
            helper = PagingHelper(tbl)
            data = helper.get_page(start, end)
            await update_widget(
                self,
                "data",
                _dmp(
                    {
                        "data": data,
                        "recordsTotal": _len,
                        "recordsFiltered": _len,
                        "length": _len,
                        "draw": draw,
                    }
                ),
            )
