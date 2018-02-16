from __future__ import absolute_import, division, print_function

from .nary import NAry


class Reduce(NAry):
    "Reduce binary modules over multiple inputs"
    def __init__(self, binary_module, left_in, right_in, outp, **kwds):
        super(Reduce, self).__init__(**kwds)
        self._binary_module = binary_module
        self._left_in = left_in
        self._right_in = right_in
        self._outp = outp
        self.binary_module_kwds = self._filter_kwds(kwds, binary_module.__init__)
        self._out_module = None

    def expand(self):
        "Expand the Reduce module into several binary modules"
        slots = [self.get_input_slot(name)
                 for name in self.inputs if name.startswith('table')]
        if len(slots) < 2:
            raise ValueError("Reduce needs at least two unputs")
        prev_slot = slots[0]
        for slot in slots[1:]:
            bin_mod = self._binary_module(**self.binary_module_kwds)
            bin_mod.input[self._left_in] = prev_slot
            bin_mod.input[self._right_in] = slot
            prev_slot = bin_mod.output[self._outp]
        self._out_module = bin_mod
        return bin_mod

    def run_step(self, run_number, step_size, howlong):
        if self._table is None:
            self._table = self._out_module.table()
        return self._return_run_step(self.state_blocked, steps_run=0)
