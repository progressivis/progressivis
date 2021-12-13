
from progressivis.table.nary import NAry
from progressivis import Scheduler


class Reduce(NAry):
    "Reduce binary modules over multiple inputs"
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
