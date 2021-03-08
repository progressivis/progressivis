"""
Dataflow Graph maintaining a graph of modules and implementing
commit/rollback semantics.
"""

import logging

from uuid import uuid4
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from progressivis.utils.toposort import toposort
from progressivis.utils.errors import ProgressiveError

logger = logging.getLogger(__name__)


class Dataflow(object):
    """Class managing a Dataflow, a configuration of modules and slots
    constructed by the user to be run by a Scheduler.

    The contents of a Dataflow can be changed at any time without
    interfering with the Scheduler. To update the Scheduler, it should
    be validated and committed first.
    """

    """
    Dataflow graph maintaining modules connected with slots.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.version = -1
        self._modules = {}
        self.inputs = {}
        self.outputs = {}
        self.valid = []
        self._slot_clashes = {}
        self.reachability = None
        # add the scheduler's dataflow into self
        self.version = scheduler.version
        for module in scheduler.modules().values():
            self._add_module(module)
        for module in scheduler.modules().values():
            for slot in module.input_slot_values():
                self.add_connection(slot)

    def clear(self):
        "Remove all the modules from the Dataflow"
        self.version = -1
        self._modules = {}
        self.inputs = {}
        self.outputs = {}
        self.valid = []
        self.reachability = None

    def generate_name(self, prefix):
        "Generate a name for a module given its class prefix."
        while True:
            name = f'{prefix}_{uuid4()}'
            if name not in self._modules:
                return name

    def modules(self):
        "Return all the modules in this dataflow"
        return self._modules.values()

    def get_visualizations(self):
        "Return the visualization modules"
        return [m.name for m in self.modules() if m.is_visualization()]

    def get_inputs(self):
        "Return the input modules"
        return [m.name for m in self.modules() if m.is_input()]

    def __delitem__(self, name):
        self.remove_module(name)

    def __getitem__(self, name):
        return self._modules.get(name, None)

    def __contains__(self, name):
        return name in self._modules

    def __len__(self):
        return len(self._modules)

    def dir(self):
        "Return the list the module names"
        return list(self._modules.keys())

    def aborted(self):
        "The dataflow has been aborted before being sent."
        # pylint: disable=protected-access
        self.clear()

    def committed(self):
        "The dataflow has been sent to the scheduler."
        self.clear()

    def add_module(self, module):
        "Add a module to this Dataflow."
        assert module.is_created()
        self._add_module(module)
        self.valid = []

    def _add_module(self, module):
        if module.name in self.inputs:
            raise ProgressiveError("Module %s already exists" % module.name)
        self._modules[module.name] = module
        self.inputs[module.name] = {}
        self.outputs[module.name] = {}

    def remove_module(self, module):
        '''Remove the specified module
           or does nothing if the module does not exist.
        '''
        if isinstance(module, str):
            module = self._modules.get(module)
        if not hasattr(module, 'name'):
            return  # module is not fully created
        # module.terminate()
        to_remove = set([module.name])
        while to_remove:
            name = to_remove.pop()
            del self._modules[name]
            self._remove_module_inputs(name)
            to_remove.update(self._remove_module_outputs(name))
        self.valid = []

    def add_connection(self, slot):
        "Declare a connection between two module slots"
        if not slot:
            return
        output_module = slot.output_module
        output_name = slot.output_name
        input_module = slot.input_module
        input_name = slot.input_name
        if input_module.input_slot_multiple(input_name):
            slot.original_name = input_name
            clashes = self._clashes(input_module, input_name)
            input_name += '.%02d.%02d' % (self.version, clashes)
            slot.input_name = input_name
            assert input_name not in self.inputs[input_module.name]
        elif input_name in self.inputs[input_module.name]:
            if slot is self.inputs[input_module.name][input_name]:
                logger.warn("redundant connection:"
                            "Input slot %s already connected to "
                            "slot %s in module %s",
                            input_name,
                            self.inputs[input_module.name][input_name],
                            input_module.name)
            else:
                raise ProgressiveError("Input slot %s already connected to"
                                       "slot %s in module %s" % (
                                           input_name,
                                           self.inputs[input_module.name][input_name],
                                           input_module.name))
        self.inputs[input_module.name][input_name] = slot
        if output_module.name not in self.outputs:
            self.outputs[output_module.name] = {output_name: [slot]}
        elif output_name not in self.outputs[output_module.name]:
            self.outputs[output_module.name][output_name] = [slot]
        else:
            self.outputs[output_module.name][output_name].append(slot)
        self.valid = []  # Not sure

    def connect(self, output_module, output_name, input_module, input_name):
        "Declare a connection between two modules slots"
        slot = output_module.create_slot(output_module, output_name,
                                         input_module, input_name)
        if not slot.validate_types():
            raise ProgressiveError('Incompatible types for slot (%s,%s) in %s',
                                   str(slot))
        self.add_connection(slot)

    def _clashes(self, module_name, input_slot_name):
        slots = self._slot_clashes.get(module_name, None)
        if slots is None:
            slots = {input_slot_name: 1}
            self._slot_clashes[module_name] = slots
            return 1  # shortcut
        slots[input_slot_name] += 1
        return slots[input_slot_name]

    def _remove_module_inputs(self, name):
        for slot in self.inputs[name].values():
            slots = self.outputs[slot.output_module.name][slot.output_name]
            nslots = [s for s in slots if s.input_module.name != name]
            assert slots != nslots  # we must remove a slot
            if nslots:
                self.outputs[slot.output_module.name][slot.output_name] = nslots
            else:
                del self.outputs[slot.output_module.name][slot.output_name]
        del self.inputs[name]

    def _remove_module_outputs(self, name):
        emptied = []
        for oslots in self.outputs[name].values():
            for slot in oslots:
                del self.inputs[slot.input_module.name][slot.input_name]
                if not self.inputs[slot.input_module.name]:
                    emptied.append(slot.input_module.name)
        del self.outputs[name]
        return emptied

    def order_modules(self, dependencies=None):
        "Compute a topological order for the modules."
        if dependencies is None:
            dependencies = self.collect_dependencies()
        runorder = toposort(dependencies)
        return runorder

    def collect_dependencies(self):
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

    def validate(self):
        '''
        Validate the Dataflow, returning [] if it is valid
        or the invalid modules otherwise.
        '''
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
    def validate_module_inputs(module, inputs):
        """Validate the input slots on a module.
        Return a list of errors, empty if no error occured.
        """
        errors = []
        for slotdesc in module.input_descriptors.values():
            slot = None
            if slotdesc.multiple:
                for islot in inputs.values():
                    if islot.original_name == slotdesc.name:
                        slot = islot
                        logger.info('Input slot "%s" renamed "%s" '
                                    'in module "%s"',
                                    islot.original_name,
                                    islot.input_name,
                                    module.name)
                        break
            else:
                slot = inputs.get(slotdesc.name)
            if slotdesc.required and slot is None:
                errors.append('Input slot "%s" missing in module "%s"' % (
                    slotdesc.name, module.name))
        return errors

    @staticmethod
    def validate_module_outputs(module, outputs):
        """Validate the output slots on a module.
        Return a list of errors, empty if no error occured.
        """
        errors = []
        for slotdesc in module.output_descriptors.values():
            slot = outputs.get(slotdesc.name)
            if slotdesc.required and slot is None:
                errors.append('Output slot "%s" missing in module "%s"' % (
                    slotdesc.name, module.name))
        return errors

    def validate_module(self, module):
        """Validate a module in the dataflow.
        Return a list of errors, empty if no error occured.
        """
        errors = self.validate_module_inputs(module,
                                             self.inputs[module.name])
        errors += self.validate_module_outputs(module,
                                               self.inputs[module.name])
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
        reachability = {inp: set(breadth_first_order(graph,
                                                     index[inp],
                                                     return_predecessors=False))
                        for inp in input_modules}
        for vis in self.get_visualizations():
            vis_index = index[vis]
            vis_reachability = set(breadth_first_order(graph.T,
                                                       vis_index,
                                                       return_predecessors=False))
            for inp in input_modules:
                inp_reachability = reachability[inp]
                if vis_index in inp_reachability:
                    inter = vis_reachability.intersection(inp_reachability)
                    inter = {k[i] for i in inter}
                    if inp in self.reachability:
                        self.reachability[inp].update(inter)
                    else:
                        self.reachability[inp] = inter
