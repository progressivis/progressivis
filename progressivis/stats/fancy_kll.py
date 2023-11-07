from __future__ import annotations

from ..core.utils import indices_len, fix_loc
from ..table.table import PTable
from ..utils.pmux import PMux
from ..core.module import Module, def_input, def_output
from ..core.decorators import process_slot, run_if_any
from datasketches import kll_floats_sketch
from .. import Slot
from functools import partial
import logging
from typing import Any, Dict, Set, List, Optional, TYPE_CHECKING
from typeguard import check_type
from collections import defaultdict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.module import ReturnRunStep


def kll_cols(slots: List[Slot]) -> Set[str]:
    result = set()
    for slot in slots:
        qry = check_type(slot.hint, Dict[str, List[float]])
        for col in qry.keys():
            result.add(col)
    return result


@def_input("table", type=PTable)
@def_output("result", PMux)
class FancyKLL(Module):
    """
    ...
    """
    def __init__(self, k: int = 200, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.kll_floats_sketch_k = partial(kll_floats_sketch, k)
        self._kll: Dict[str, kll_floats_sketch] = defaultdict(self.kll_floats_sketch_k)
        self._kll_cols: Optional[Set[str]] = None
        self.default_step_size: int = 10000
        self.result = PMux()

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.result.clear()
        self._kll = {}

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            indices = ctx.table.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if not steps:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = ctx.table.data()
            slots = self.get_slots_connected_to("result")
            if self._kll_cols is None:
                self._kll_cols = kll_cols(slots)
            steps *= len(self._kll_cols)
            for col in self._kll_cols:
                kll = self._kll[col]
                column = input_df[col]
                column = column.loc[fix_loc(indices)]
                sk = self.kll_floats_sketch_k()
                sk.update(column)
                kll.merge(sk)
            # each slot must have a hint like:
            # {col1: [10, 20], col2: [40, 50]}
            for slot in slots:
                key = slot.mux_key
                res = {col: self._kll[col].get_quantiles(quants) for col, quants in slot.hint.items()}
                self.result.update(key, res)
            return self._return_run_step(self.next_state(ctx.table), steps)
