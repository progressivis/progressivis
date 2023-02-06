from __future__ import annotations

from progressivis import ProgressiveError
from ..core.module import Module
from progressivis.core.module import ReturnRunStep, JSon, def_output
from ..utils.psdict import PDict

from typing import Dict, Any, Optional


@def_output("result", PDict)
class DynVar(Module):
    def __init__(
        self,
        init_val: Optional[Dict[str, Any]] = None,
        translation: Optional[Dict[str, Any]] = None,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self.tags.add(self.TAG_INPUT)
        self._has_input = False
        if not (translation is None or isinstance(translation, dict)):
            raise ProgressiveError("translation must be a dictionary")
        self._translation = translation
        if not (init_val is None or isinstance(init_val, dict)):
            raise ProgressiveError("init_val must be a dictionary")
        self.result = PDict({} if init_val is None else init_val)

    def is_input(self) -> bool:
        return True

    def has_input(self) -> bool:
        return self._has_input

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        return self._return_run_step(self.state_blocked, steps_run=1)

    def predict_step_size(self, duration: float) -> int:
        return 1

    async def from_input(self, input_: JSon) -> str:
        if not isinstance(input_, dict):
            raise ProgressiveError("Expecting a dictionary")
        last = PDict(self.result)  # shallow copy
        values = input_
        if self._translation is not None:
            res = {}
            for k, v in values.items():
                for syn in self._translation[k]:
                    res[syn] = v
            values = res
        for (k, v) in input_.items():
            last[k] = v
        await self.scheduler().for_input(self)
        if self.result is None:
            self.result = PDict(values)
        else:
            self.result.update(values)
        self._has_input = True
        return ""
