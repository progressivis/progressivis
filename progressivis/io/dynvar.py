from progressivis import ProgressiveError
from ..table.module import TableModule
from ..utils.psdict import PsDict


class DynVar(TableModule):
    def __init__(self, init_val=None, translation=None, **kwds):
        super().__init__(**kwds)
        self._has_input = False
        self.prioritize = set()
        if not (translation is None or isinstance(translation, dict)):
            raise ProgressiveError("translation must be a dictionary")
        self._translation = translation
        if not (init_val is None or isinstance(init_val, dict)):
            raise ProgressiveError("init_val must be a dictionary")
        self.result = PsDict({} if init_val is None else init_val)

    def is_input(self):
        return True

    def has_input(self):
        return self._has_input

    def predict_step_size(self, duration):
        return 1

    def run_step(self, run_number, step_size, howlong):
        return self._return_run_step(self.state_blocked, steps_run=1)
        # raise StopIteration()

    async def from_input(self, input_):
        if not isinstance(input_, dict):
            raise ProgressiveError("Expecting a dictionary")
        last = PsDict(self.result)  # shallow copy
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
        self.result.update(values)
        self._has_input = True
        return ""
