"""
Dataflow Graph maintaining a graph of modules and implementing
commit/rollback semantics.
"""
from __future__ import annotations

from typing import Any, Dict, Set, List, TYPE_CHECKING, Optional, Union

import logging

# import pprint

from uuid import uuid4
from scipy.sparse import csr_matrix  # type: ignore
from scipy.sparse.csgraph import breadth_first_order  # type: ignore

from progressivis.utils.toposort import toposort
from progressivis.utils.errors import ProgressiveError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .scheduler import Scheduler
    from .module import Module
    from .slot import Slot

Dependencies = Dict[str, Set[str]]
Order = List[str]


class Dataflow:
    """Class managing a Dataflow, a configuration of modules and slots
    constructed by the user to be run by a Scheduler.

    The contents of a Dataflow can be changed at any time without
    interfering with the Scheduler. To update the Scheduler, the Dataflow
    should be validated first then commited.
    """

    multiple_slots_name_generator = 1

    def __init__(self, scheduler: Scheduler):
        self.scheduler: Scheduler = scheduler
        """Scheduler associated with this Dataflow"""
        self._modules: Dict[str, Module] = {}
        self.inputs: Dict[str, Dict[str, Slot]] = {}
        self.outputs: Dict[str, Dict[str, List[Slot]]] = {}
        self.valid: List[Module] = []
        self.reachability: Dict[str, Any] = {}
        # add the scheduler's dataflow into self
        self.version: int = scheduler.version
        for module in scheduler.modules().values():
            self._add_module(module)
        for module in scheduler.modules().values():
            for slot in module.input_slot_values():
                self.add_connection(slot, rename=False)

    def clear(self) -> None:
        "Remove all the modules from the Dataflow"
        self.version = -1
        self._modules = {}
        self.inputs = {}
        self.outputs = {}
        self.valid = []
        self.reachability = {}

    def generate_name(self, prefix: str) -> str:
        "Generate a name for a module given its class prefix."
        for i in range(1, 10):
            mid = f"{prefix}_{i}"
            if mid not in self._modules:
                return mid
        return f"{prefix}_{uuid4()}"

    def modules(self) -> Dict[str, Module]:
        "Return all the modules in this dataflow"
        return self._modules

    def group_modules(self, *names: str) -> List[str]:
        nameset = set(names)
        if not nameset:
            return []
        return [mod.name for mod in self.modules().values() if mod.group in nameset]

    def dir(self) -> List[str]:
        "Return the name of all the modules"
        return list(self._modules.keys())

    def get_visualizations(self) -> List[str]:
        "Return the visualization modules"
        return [m.name for m in self.modules().values() if m.is_visualization()]

    def get_inputs(self) -> List[str]:
        "Return the input modules"
        return [m.name for m in self.modules().values() if m.is_input()]

    def __delitem__(self, name: str) -> None:
        self.delete_modules(name)
        # raise RuntimeError("Cannot delete a module directly, use delete_modules")
        # self._remove_module(self._modules[name])

    def __getitem__(self, name: str) -> Module:
        return self._modules[name]

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    def groups(self) -> Set[str]:
        return {mod.group for mod in self.modules().values() if mod.group is not None}

    def aborted(self) -> None:
        "The dataflow has been aborted before being sent."
        # pylint: disable=protected-access
        self.clear()

    def committed(self) -> None:
        "The dataflow has been sent to the scheduler."
        self.clear()

    def add_module(self, module: Module) -> None:
        "Add a module to this Dataflow."
        assert module.is_created()
        self._add_module(module)
        self.valid = []

    def _add_module(self, module: Module) -> None:
        if module.name in self.inputs:
            raise ProgressiveError("Module %s already exists" % module.name)
        self._modules[module.name] = module
        self.inputs[module.name] = {}
        self.outputs[module.name] = {}

    def _remove_module(self, module: Module) -> None:
        """Remove the specified module
        or does nothing if the module does not exist.
        """
        # if isinstance(module, str):
        #     module = self._modules.get(module)
        if not hasattr(module, "name"):
            return  # module is not fully created
        # module.terminate()
        name = module.name
        del self._modules[name]
        self._remove_module_inputs(name)
        self._remove_module_outputs(name)
        self.valid = []

    def add_connection(self, slot: Optional[Slot], rename: bool = True) -> None:
        "Declare a connection between two module slots"
        if not slot:
            return
        output_module = slot.output_module
        output_name = slot.output_name
        input_module = slot.input_module
        input_name = slot.original_name or slot.input_name
        if input_module is None:
            return
        assert input_name is not None
        if input_module.input_slot_multiple(input_name):
            if rename:
                slot.original_name = input_name
                input_name += f".{self.multiple_slots_name_generator:04}"
                self.multiple_slots_name_generator += 1
                logger.info(f"{slot.original_name} renamed {input_name}")
                slot.input_name = input_name
            else:
                input_name = slot.input_name
        if input_name in self.inputs[input_module.name]:
            if slot is self.inputs[input_module.name][input_name]:
                logger.warn(
                    "redundant connection:"
                    "Input slot %s already connected to "
                    "slot %s in module %s",
                    input_name,
                    self.inputs[input_module.name][input_name],
                    input_module.name,
                )
            else:
                raise ProgressiveError(
                    "Input slot %s already connected to"
                    "slot %s in module %s"
                    % (
                        input_name,
                        self.inputs[input_module.name][input_name],
                        input_module.name,
                    )
                )
        assert input_name is not None
        self.inputs[input_module.name][input_name] = slot
        if output_module.name not in self.outputs:
            self.outputs[output_module.name] = {output_name: [slot]}
        elif output_name not in self.outputs[output_module.name]:
            self.outputs[output_module.name][output_name] = [slot]
        else:
            self.outputs[output_module.name][output_name].append(slot)
        self.valid = []  # Not sure

    def connect(
        self,
        output_module: Module,
        output_name: str,
        input_module: Module,
        input_name: str,
    ) -> None:
        "Declare a connection between two modules slots"
        slot = output_module.create_slot(output_name, input_module, input_name)
        if not slot.validate_types():
            raise ProgressiveError(
                "Incompatible types for slot (%s,%s) in %s"
                % (output_name, input_name, str(slot))
            )
        self.add_connection(slot)

    def _remove_module_inputs(self, name: str) -> None:
        for slot in self.inputs[name].values():
            outname = slot.output_name
            slots = self.outputs[slot.output_module.name][outname]
            nslots = [
                s for s in slots if s.input_module and s.input_module.name != name
            ]
            # assert slots != nslots  # we must remove a slot
            if nslots:
                self.outputs[slot.output_module.name][outname] = nslots
            else:
                del self.outputs[slot.output_module.name][outname]
        del self.inputs[name]

    def _remove_module_outputs(self, name: str) -> None:
        for oslots in self.outputs[name].values():
            for slot in oslots:
                assert slot.input_module is not None and slot.input_name is not None
                del self.inputs[slot.input_module.name][slot.input_name]
                # if not self.inputs[slot.input_module.name]:
                #    del self.inputs[slot.input_module.name]
                #     emptied.append(slot.input_module.name)
                # if the module become invalid, it should be removed
                # Equivalent of SIGCHILD
                # elif self.validate_module(slot.input_module):
                #     emptied.append(slot.input_module.name)
        del self.outputs[name]
        # return emptied

    def order_modules(self, dependencies: Optional[Dependencies] = None) -> Order:
        "Compute a topological order for the modules."
        if dependencies is None:
            dependencies = self.collect_dependencies()
        runorder = toposort(dependencies)
        return runorder

    def collect_dependencies(self) -> Dict[str, Set[str]]:
        "Return the dependecies of the modules"
        errors = self.validate()
        if errors:
            raise ProgressiveError(f"Invalid dataflow: {errors}")
        dependencies = {}
        for valid in self.valid:
            module = valid.name
            slots = self.inputs[module]
            outs = [m.output_module.name for m in slots.values()]
            dependencies[module] = set(outs)
        return dependencies

    def validate(self) -> List[str]:
        """
        Validate the Dataflow, returning [] if it is valid
        or the invalid modules otherwise.
        """
        errors = []
        if not self.valid:
            valid = []
            for module in self._modules.values():
                error = self.validate_module(module)
                if error:
                    errors += error
                else:
                    valid.append(module)
            self.valid = valid
        for module in self.valid:
            module.validate()
        return errors

    @staticmethod
    def validate_module_inputs(module: Module, inputs: Dict[str, Slot]) -> List[str]:
        """Validate the input slots on a module.
        Return a list of errors, empty if no error occured.
        """
        errors = []
        inputs = dict(inputs)
        for slotdesc in module.input_descriptors.values():
            slot = None
            if slotdesc.multiple:
                for islot in list(inputs.values()):
                    if islot.original_name == slotdesc.name:
                        slot = islot
                        if islot.original_name != islot.input_name:
                            logger.info(
                                f'Input slot "{islot.original_name}" '
                                f'renamed "{islot.input_name}" '
                                f'in module "{module.name}"'
                            )
                        assert slot.input_name is not None and inputs.pop(
                            slot.input_name, False
                        ), f"Input slot {slot.input_name} not in {inputs}"
                    # Iterate over all the inputs to remove the multiple slots
            else:
                slot = inputs.get(slotdesc.name)
                if slot:
                    assert slot.input_name is not None
                    inputs.pop(slot.input_name)
            if slotdesc.required and slot is None:
                errors.append(
                    f'Input slot "{slotdesc.name}" missing in module "{module.name}"'
                )
        if inputs:
            errors.append(
                f"Invalid input slot(s) {list(inputs.keys())} for module {module.name}"
            )
        return errors

    @staticmethod
    def validate_module_outputs(
        module: Module, outputs: Dict[str, List[Slot]]
    ) -> List[str]:
        """Validate the output slots on a module.
        Return a list of errors, empty if no error occured.
        """
        errors: List[str] = []
        outputs = dict(outputs)
        for slotdesc in module.output_descriptors.values():
            slot = outputs.get(slotdesc.name)
            if slotdesc.required and slot is None:
                errors.append(
                    'Output slot "%s" missing in module "%s"'
                    % (slotdesc.name, module.name)
                )
            if slot:
                del outputs[slotdesc.name]
        if outputs:
            errors.append(
                f"Invalid output slot(s) {list(outputs.keys())} for module {module.name}"
            )
        return errors

    def validate_module(self, module: Module) -> List[str]:
        """Validate a module in the dataflow.
        Return a list of errors, empty if no error occured.
        """
        errors = self.validate_module_inputs(module, self.inputs[module.name])
        errors += self.validate_module_outputs(module, self.outputs[module.name])
        return errors

    @staticmethod
    def _dependency_csgraph(
        dependencies: Dict[str, Dict[str, Slot]], index: Dict[str, int]
    ) -> Any:
        size = len(index)
        row = []
        col = []
        data = []
        for vertex1, vertices in dependencies.items():
            for vertex2 in vertices.values():
                col.append(index[vertex1])
                row.append(index[vertex2.output_module.name])
                data.append(1)
        return csr_matrix((data, (row, col)), shape=(size, size))

    def _compute_reachability(self, dependencies: Dict[str, Dict[str, Slot]]) -> None:
        if self.reachability:
            return
        input_modules = self.get_inputs()
        k = list(dependencies.keys())
        index = dict(zip(k, range(len(k))))
        graph = self._dependency_csgraph(dependencies, index)
        self.reachability = {}
        reachability = {
            inp: set(breadth_first_order(graph, index[inp], return_predecessors=False))
            for inp in input_modules
        }
        for vis in self.get_visualizations():
            vis_index = index[vis]
            vis_reachability = set(
                breadth_first_order(graph.T, vis_index, return_predecessors=False)
            )
            for inp in input_modules:
                inp_reachability = reachability[inp]
                if vis_index in inp_reachability:
                    inter = vis_reachability.intersection(inp_reachability)
                    inter = {k[i] for i in inter}
                    if inp in self.reachability:
                        self.reachability[inp].update(inter)
                    else:
                        self.reachability[inp] = inter

    def collateral_damage(self, *names: str) -> Set[str]:
        """Return the list of modules deleted when the specified one is deleted.

        :param name: module to delete
        :returns: list of modules relying on or feeding the specified module
        :rtype: set

        """
        deps = set()  # modules connected with a required slot

        queue = set(names)
        queue &= self._modules.keys()
        # TODO check if all the modules exist
        done: Set[str] = set()

        while queue:
            name = queue.pop()
            done.add(name)
            deps.add(name)
            # collect children and ancestors
            self._collect_collaterals(name, deps)
            queue.update(deps - done)
        return deps

    def _collect_collaterals(self, name: str, deps: Set[str]) -> None:
        self._collect_output_collaterals(name, deps)
        self._collect_input_collaterals(name, deps)

    def _collect_output_collaterals(self, name: str, deps: Set[str]) -> None:
        outputs = self.outputs[name]
        for olist in outputs.values():
            if olist is None:
                continue
            for oslot in olist:
                self._collect_input_collaterals_if_required(oslot, deps)

    def _collect_input_collaterals_if_required(
        self, oslot: Slot, deps: Set[str]
    ) -> None:
        # collect input module only if the input slot being removed is required
        module = oslot.input_module
        assert module is not None
        if module.name in deps:
            return  # already being removed
        slot_name = oslot.original_name or oslot.input_name
        assert slot_name is not None
        desc = module.input_slot_descriptor(slot_name)
        if desc.required:
            deps.add(module.name)

    def _collect_output_collaterals_if_required(
        self, islot: Slot, deps: Set[str]
    ) -> None:
        module = islot.output_module
        if module.name in deps:  # already being removed
            return
        slot_name = islot.output_name
        olist = self.outputs[module.name][slot_name]  # the slot list exists
        desc = module.output_slot_descriptor(slot_name)
        if not desc.required:
            return
        # if the slot is required, the module is removed if all it outputs are
        remaining = len(olist)
        for oslot in olist:
            if oslot.input_module and oslot.input_module.name in deps:
                remaining -= 1
        if remaining == 0:
            deps.add(module.name)

    def _collect_input_collaterals(self, name: str, deps: Set[str]) -> None:
        inputs = self.inputs[name]
        for islot in inputs.values():
            if islot is None:
                continue
            self._collect_output_collaterals_if_required(islot, deps)

    def _test_modules_removed(self, names: Set[str]) -> List[str]:
        errors: List[str] = []
        for iname in self.inputs:
            if iname in names:
                errors.append(f"Input name {iname} in deleted names")
            for islot in self.inputs[iname].values():
                if islot.input_module and islot.input_module.name in names:
                    errors.append(
                        f"Input slot {iname} points to deleted names: {islot}"
                    )
                if islot.output_module.name in names:
                    errors.append(
                        f"Input slot {iname} points to deleted names: {islot}"
                    )
        for oname in self.outputs:
            if oname in names:
                errors.append(f"Output name {oname} in deleted names")
            for oslots in self.outputs[oname].values():
                for oslot in oslots:
                    if oslot.input_module and oslot.input_module.name in names:
                        errors.append(
                            f"Output slot {oname} points to deleted names: {oslot}"
                        )
                    if oslot.output_module.name in names:
                        errors.append(
                            f"Output slot {oname} points to deleted names: {oslot}"
                        )
        return errors

    def delete_modules(self, *name_or_mod: Union[str, Module]) -> None:
        names = {m if isinstance(m, str) else m.name for m in name_or_mod}
        collaterals = self.collateral_damage(*names)
        if collaterals != names:
            raise ValueError(
                f"pass the result of collateral_damage ({collaterals})"
                "for the specified modules"
            )
        for mod in names:
            self._remove_module(self[mod])
        # errs = self._test_modules_removed(names)
        # if errs:
        #     pprint.pprint(errs)

    def die_if_deps_die(
        self, name: str, deps: Set[str], maybe_deps: Set[str]
    ) -> Optional[bool]:
        """Return True if the module would die if the deps
        modules die, False if not, None if not sure.

        :param deps: a set of module names that will die
        :param maybe_deps: a set of module names that could die
        :returns: True if the module dies, False if it does not,
          None if not sure
        :rtype: Boolean or None

        """
        ret: Optional[bool] = False
        inputs = self.inputs[name]
        imods = {islot.output_module.name for islot in inputs.values()}
        if imods.issubset(deps):  # all input will be deleted, we die
            return True
        if imods.issubset(deps | maybe_deps):
            ret = None  # Maybe
        outputs = self.outputs[name]
        omods = {
            oslot.input_module.name
            for oslots in outputs.values()
            for oslot in oslots
            if oslot.input_module is not None
        }
        if omods.issubset(deps):
            return True
        if omods.issubset(deps | maybe_deps):
            ret = None
        return ret
