"""
Dataflow Graph maintaining a graph of modules and implementing
commit/rollback semantics.
"""
from __future__ import absolute_import, division, print_function

import logging

from uuid import uuid4
import six

from .scheduler import Scheduler
from .toposort import toposort

logger = logging.getLogger(__name__)

class Dataflow(object):
    """Class managing a Dataflow, a configuration of modules and slots
    constructed by the user to be run by a Scheduler.

    The contents of a Dataflow can be changed at any time without
    interfering with the Scheduler. To update the Scheduler, it should
    be validated and committed first.
    """
    default = None
    """
    Dataflow graph maintaining modules connected with slots.
    """
    def __init__(self, scheduler=None):
        if scheduler is None:
            scheduler = Scheduler.default
        assert scheduler is not None
        self.scheduler = scheduler
        self._modules = {}
        self._inputs = {}
        self._outputs = {}
        self.valid = []

    def clear(self):
        "Remove all the modules from the Dataflow"
        self._modules = {}
        self._inputs = {}
        self._outputs = {}
        self.valid = []

    def generate_name(self, prefix):
        "Generate a name for a module given its class prefix."
        for i in range(1, 10):
            mid = '%s_%d' % (prefix, i)
            if mid not in self._modules:
                return mid
        return '%s_%s' % (prefix, uuid4())

    def __getitem__(self, name):
        return self._modules[name]

    def __contains__(self, name):
        return name in self._modules

    def dir(self):
        "Return the list of the module names"
        return list(self._modules.keys())

    def add_scheduler(self, scheduler=None):
        "Fill-up this Dataflow with the dataflow run by a specified Scheduler"
        if scheduler is None:
            scheduler = self.scheduler
        for module in scheduler.modules().values():
            self._add_module(module)
        for module in scheduler.modules().values():
            for slot in module.output_slots_values():
                self.add_connection(slot)

    def add_module(self, module):
        "Add a module to this Dataflow."
        assert module.is_created()
        self._add_module(module)
        self.valid = []

    def _add_module(self, module):
        assert module.name not in self._inputs
        self._modules[module.name] = module
        self._inputs[module.name] = {}
        self._outputs[module.name] = {}

    def remove_module(self, module):
        "Remove the specified module"
        if isinstance(module, six.string_types):
            module = self._modules[module]
        # module.terminate()
        del self._modules[module.name]
        self._remove_module_inputs(module.name)
        self._remove_module_outputs(module.name)
        self.valid = []

    def add_connection(self, slot):
        "Declare a connection between two module slots"
        output_module = slot.output_module
        output_name = slot.output_name
        input_module = slot.input_module
        input_name = slot.input_name
        assert input_name not in self._inputs[input_module.name]
        self._inputs[input_module.name][input_name] = slot
        if output_module.name not in self._outputs:
            self._outputs[output_module.name] = {output_name: [slot]}
        elif output_name not in self._outputs[output_module.name]:
            self._outputs[output_module.name][output_name] = [slot]
        else:
            self._outputs[output_module.name].append(slot)
        self.valid = [] # Not sure

    def _remove_module_inputs(self, name):
        for slot in self._inputs[name].values():
            slots = self._outputs[slot.output_module.name][slot.output_name]
            nslots = [s for s in slots if s.output_module.name != name]
            if nslots:
                self._outputs[slot.output_module.name][slot.output_name] = nslots
            else:
                del self._outputs[slot.output_module.name][slot.output_name]
        del self._inputs[name]

    def _remove_module_outputs(self, name):
        module_slots = self._outputs[name]
        for (sname, slots) in module_slots.items():
            nslots = [s for s in slots if s.input_module.name != name]
            if nslots == slots:
                continue # no need to change, weird
            elif nslots:
                module_slots[sname] = nslots
            else:
                del module_slots[sname]
        del self._outputs[name]

    def collect_dependencies(self):
        "Return the dependecies of the modules"
        self.validate()
        dependencies = {}
        for valid in self.valid:
            module = valid.name
            slots = self._inputs[module]
            outs = [m.output_module.name for m in slots.values()]
            dependencies[module] = set(outs)
        return dependencies

    def order_modules(self):
        """Compute a topological order for the modules.
        """
        dependencies = self.collect_dependencies()
        runorder = toposort(dependencies)
        return runorder

    def validate(self):
        "Validate the Dataflow, returning [] if it is valid or the invalid modules otherwise."
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
        return errors

    @staticmethod
    def validate_module_inputs(module, inputs):
        """Validate the input slots on a module.
        Return a list of errors, empty if no error occured.
        """
        errors = []
        for slotdesc in module.input_descriptors.values():
            slot = inputs.get(slotdesc.name)
            if slotdesc.required and slot is None:
                errors.append('Input slot "%s" missing in module "%s"'%(
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
                errors.append('Output slot "%s" missing in module "%s"'%(
                    slotdesc.name, module.name))
        return errors


    def validate_module(self, module):
        """Validate a module in the dataflow.
        Return a list of errors, empty if no error occured.
        """
        errors = self.validate_module_inputs(module, self._inputs[module.name])
        errors += self.validate_module_outputs(module, self._inputs[module.name])
        return errors

    def __len__(self):
        return len(self._modules)

Dataflow.default = Dataflow()
