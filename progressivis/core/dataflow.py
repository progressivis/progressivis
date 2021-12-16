"""
Dataflow Graph maintaining a graph of modules and implementing
commit/rollback semantics.
"""
from __future__ import annotations

from typing import Any, Dict, Set, List, TYPE_CHECKING, Optional

import logging

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
    interfering with the Scheduler. To update the Scheduler, it should
    be validated and committed first.
    """

    """
    Dataflow graph maintaining modules connected with slots.
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self._modules: Dict[str, Module] = {}
        self.inputs: Dict[str, Dict] = {}
        self.outputs: Dict[str, Dict] = {}
        self.valid: List[Module] = []
        self._slot_clashes: Dict[str, Dict[str, int]] = {}
        self.reachability: Dict[str, Any] = {}
        # add the scheduler's dataflow into self
        self.version: int = scheduler.version
        for module in scheduler.modules().values():
            self._add_module(module)
        for module in scheduler.modules().values():
            for slot in module.input_slot_values():
                self.add_connection(slot)

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

    def modules(self) -> List[Module]:
        "Return all the modules in this dataflow"
        return list(self._modules.values())

    def dir(self) -> List[str]:
        "Return the name of all the modules"
        return list(self._modules.keys())

    def get_visualizations(self) -> List[str]:
        "Return the visualization modules"
        return [m.name for m in self.modules() if m.is_visualization()]

    def get_inputs(self) -> List[str]:
        "Return the input modules"
        return [m.name for m in self.modules() if m.is_input()]

    def __delitem__(self, name: str) -> None:
        self.remove_module(self._modules[name])

    def __getitem__(self, name: str) -> Module:
        return self._modules[name]

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __len__(self) -> int:
        return len(self._modules)

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

    def remove_module(self, module: Module) -> None:
        """Remove the specified module
           or does nothing if the module does not exist.
        """
        # if isinstance(module, str):
        #     module = self._modules.get(module)
        if not hasattr(module, "name"):
            return  # module is not fully created
        # module.terminate()
        to_remove = set([module.name])
        while to_remove:
            name = to_remove.pop()
            del self._modules[name]
            self._remove_module_inputs(name)
            to_remove.update(self._remove_module_outputs(name))
        self.valid = []

    def add_connection(self, slot: Slot) -> None:
        "Declare a connection between two module slots"
        if not slot:
            return
        output_module = slot.output_module
        output_name = slot.output_name
        input_module = slot.input_module
        input_name = slot.input_name
        if input_module.input_slot_multiple(input_name):
            slot.original_name = input_name
            clashes = self._clashes(input_module.name, input_name)
            input_name += f".{self.version:02d}.{clashes:02d}"
            slot.input_name = input_name
            assert input_name not in self.inputs[input_module.name]
        elif input_name in self.inputs[input_module.name]:
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
            input_name: str) -> None:
        "Declare a connection between two modules slots"
        slot = output_module.create_slot(
            output_name, input_module, input_name
        )
        if not slot.validate_types():
            raise ProgressiveError(
                "Incompatible types for slot (%s,%s) in %s" % (
                    output_name,
                    input_name,
                    str(slot)
                )
            )
        self.add_connection(slot)

    def _clashes(self, module_name: str, input_slot_name: str) -> int:
        slots = self._slot_clashes.get(module_name, None)
        if slots is None:
            slots = {input_slot_name: 1}
            self._slot_clashes[module_name] = slots
            return 1  # shortcut
        slots[input_slot_name] += 1
        return slots[input_slot_name]

    def _remove_module_inputs(self, name: str):
        for slot in self.inputs[name].values():
            outname = slot.output_name
            slots = self.outputs[slot.output_module.name][outname]
            nslots = [s for s in slots if s.input_module.name != name]
            assert slots != nslots  # we must remove a slot
            if nslots:
                self.outputs[slot.output_module.name][outname] = nslots
            else:
                del self.outputs[slot.output_module.name][outname]
        del self.inputs[name]

    def _remove_module_outputs(self, name: str):
        emptied = []
        for oslots in self.outputs[name].values():
            for slot in oslots:
                del self.inputs[slot.input_module.name][slot.input_name]
                if not self.inputs[slot.input_module.name]:
                    emptied.append(slot.input_module.name)
                # if the module become invalid, it should be removed
                # Equivalent of SIGCHILD
                elif self.validate_module(slot.input_module):
                    emptied.append(slot.input_module.name)
        del self.outputs[name]
        return emptied

    def order_modules(self, dependencies : Dependencies = None) -> Order:
        "Compute a topological order for the modules."
        if dependencies is None:
            dependencies = self.collect_dependencies()
        runorder = toposort(dependencies)
        return runorder

    def collect_dependencies(self) -> Dict[str, Set[str]]:
        "Return the dependecies of the modules"
        errors = self.validate()
        if errors:
            raise ProgressiveError("Invalid dataflow", errors)
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
    def validate_module_inputs(module, inputs: Dict[str, Slot]) -> List[str]:
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
                        logger.info(
                            'Input slot "%s" renamed "%s" ' 'in module "%s"',
                            islot.original_name,
                            islot.input_name,
                            module.name,
                        )
                        inputs.pop(slot.input_name)
                        # Iterate over all the inputs to remove the multiple slots
            else:
                slot = inputs.get(slotdesc.name)
                if slot:
                    inputs.pop(slot.input_name)
            if slotdesc.required and slot is None:
                errors.append(
                        f'Input slot "{slotdesc.name}" missing in module "{module.name}"'
                )
        if inputs:
            errors.append(
                f'Invalid input slot(s) {list(inputs.keys())} for module {module.name}'
            )
        return errors

    @staticmethod
    def validate_module_outputs(module, outputs: Dict[str, List[Slot]]):
        """Validate the output slots on a module.
        Return a list of errors, empty if no error occured.
        """
        errors = []
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
                f'Invalid output slot(s) {list(outputs.keys())} for module {module.name}'
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
    def _dependency_csgraph(dependencies, index):
        size = len(index)
        row = []
        col = []
        data = []
        for (vertex1, vertices) in dependencies.items():
            for vertex2 in vertices.values():
                col.append(index[vertex1])
                row.append(index[vertex2.output_module.name])
                data.append(1)
        return csr_matrix((data, (row, col)), shape=(size, size))

    def _compute_reachability(self, dependencies):
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

    def collateral_damage(self, name: str) -> Set[str]:
        """Return the list of modules deleted when the specified one is deleted.

        :param name: module to delete
        :returns: list of modules relying on or feeding the specified module
        :rtype: set

        """
        assert isinstance(name, str)

        if name not in self._modules:
            return Set()
        deps = set([name])  # modules connected with a required slot
        maybe_deps: Set[str] = set()  # modules with a non required one
        queue = set(deps)
        done: Set[str] = set()

        while queue:
            name = queue.pop()
            done.add(name)
            # collect children and ancestors
            self[name].collect_deps(deps, maybe_deps)
            queue = deps - done

        # Check from the maybe_deps if some would be deleted for sure
        again = True
        while again:
            again = False
            for maybe in maybe_deps:
                die = self[name].die_if_deps_die(deps, maybe_deps)
                if die:
                    deps.add(name)
                    maybe_deps.remove(name)
                elif die is None:
                    again = True  # need to iterate
                else:
                    maybe_deps.remove(name)
        return deps

    def _collect_deps(self,
                      name: str,
                      deps: Set[str],
                      maybe_deps: Set[str]) -> None:
        self._input_deps(name, deps, maybe_deps)
        self._output_deps(name, deps, maybe_deps)

    def _input_deps(self,
                    name: str,
                    deps: Set[str],
                    maybe_deps: Set[str]) -> None:
        mod = self[name]
        for olist in mod.output_slot_values():
            if olist is None:
                continue
            for oslot in olist:
                module = oslot.input_module
                if module.name in deps:
                    continue
                slot_name = oslot.input_name
                desc = module.input_slot_descriptor(slot_name)
                if desc.required:
                    deps.add(module.name)
                    maybe_deps.discard(module.name)  # in case
                else:
                    maybe_deps.add(module.name)

    def _output_deps(self,
                     name: str,
                     deps: Set[str],
                     maybe_deps: Set[str]) -> None:
        mod = self[name]
        for islot in mod.input_slot_values():
            if islot is None:
                continue
            module = islot.output_module
            if module.name in deps:
                continue
            slot_name = islot.output_name
            desc = module.output_slot_descriptor(slot_name)
            if desc.required:
                deps.add(module.name)
                maybe_deps.discard(module.name)  # in case
            else:
                maybe_deps.add(module.name)

    def die_if_deps_die(self,
                        name: str,
                        deps: Set[str],
                        maybe_deps: Set[str]) -> Optional[bool]:
        """Return True if the module would die if the deps
        modules die, False if not, None if not sure.

        :param deps: a set of module names that will die
        :param maybe_deps: a set of module names that could die
        :returns: True if the module dies, False if it does not,
          None if not sure
        :rtype: Boolean or None

        """
        ret: Optional[bool] = False
        mod = self[name]
        imods = {islot.output_module.name
                 for islot in mod.input_slot_values()}
        if imods.issubset(deps):  # all input will be deleted, we die
            return True
        if imods.issubset(deps | maybe_deps):
            ret = None  # Maybe
        omods = {oslot.input_module.name
                 for oslots in mod.output_slot_values()
                 for oslot in oslots}
        if omods.issubset(deps):
            return True
        if omods.issubset(deps | maybe_deps):
            ret = None
        return ret
