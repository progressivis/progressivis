
from .nary import NAry
from progressivis import Scheduler


class Reduce(NAry):
    "Reduce binary modules over multiple inputs"
    # def __init__(self, binary_module, left_in, right_in, outp, **kwds):
    #     super(Reduce, self).__init__(**kwds)
    #     self._binary_module = binary_module
    #     self._left_in = left_in
    #     self._right_in = right_in
    #     self._outp = outp
    #     self.binary_module_kwds = self._filter_kwds(kwds,
    #                                                 binary_module.__init__)
    #     self.out_module = None

    # def _expand(self):
    #     "Expand the Reduce module into several binary modules"
    #     slots = self.get_input_slot_multiple()
    #     if len(slots) < 2:
    #         raise ValueError("Reduce needs at least two unputs")
    #     prev_slot = slots[0]
    #     with self.scheduler():
    #         for slot in slots[1:]:
    #             bin_mod = self._binary_module(**self.binary_module_kwds)
    #             bin_mod.input[self._left_in] = prev_slot
    #             bin_mod.input[self._right_in] = slot
    #             prev_slot = bin_mod.output[self._outp]
    #     self.out_module = bin_mod
    #     return bin_mod

    # def run_step(self, run_number, step_size, howlong):
    #     if self.out_module is None:
    #         self._expand()
    #     if self._table is None:
    #         self._table = self._out_module.table()
    #     return self._return_run_step(self.state_blocked, steps_run=0)

    @staticmethod
    def expand(binary_module, left_in, right_in, outp, slots,
               **binary_module_kwds):
        if len(slots) < 2:
            raise ValueError("Reduce needs at least two unputs")
        scheduler = binary_module_kwds.get("scheduler")
        if scheduler is None:
            scheduler = Scheduler.default
            binary_module_kwds["scheduler"] = scheduler
        prev_slot = slots[0]
        with scheduler:
            for slot in slots[1:]:
                bin_mod = binary_module(**binary_module_kwds)
                bin_mod.input[left_in] = prev_slot
                bin_mod.input[right_in] = slot
                prev_slot = bin_mod.output[outp]
        return bin_mod
