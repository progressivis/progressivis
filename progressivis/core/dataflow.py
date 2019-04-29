"""
Dataflow Graph maintaining a graph of modules and implementing
commit/rollback semantics.
"""
from __future__ import absolute_import, division, print_function

import logging

from uuid import uuid4
import six

from .utils import ProgressiveError
from .scheduler_base import BaseScheduler

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
            scheduler = BaseScheduler.default
        assert scheduler is not None
        self.scheduler = scheduler
        self._modules = dict()
        self._inputs = dict()
        self._outputs = dict()

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

    def add_module(self, module):
        "Add a module to this Dataflow."
        assert module.is_created()
        assert module.name not in self._inputs
        assert module.name not in self._outputs
        self._modules[module.name] = module
        self._inputs[module.name] = {}
        self._outputs[module.name] = {}

    def remove_module(self, module):
        "Remove the specified module"
        if isinstance(module, six.string_types):
            module = self._modules[module]
        module.terminate()
        del self._modules[module.name]
        del self._inputs[module.name]
        del self._outputs[module.name]

    def add_connection(self, output_module, output_name,
                       input_module, input_name):
        "Declares a connection between two module slots"
        assert input_name not in self._inputs[input_module.name]
        assert output_name not in self._outputs[output_module.name]
        self._inputs[input_module.name][input_name] = (output_module.name, output_name)
        self._outputs[output_module.name][output_name] = (input_module.name, input_name)

    def collect_dependencies(self, only_required=False):
        "Return the dependecies of the modules"
        dependencies = {}
        for (mid, module) in six.iteritems(self._modules):
            if not module.is_valid():
                continue
            outs = [m.output_module.name for m in module.input_slot_values()
                    if m and (not only_required or
                              module.input_slot_required(m.input_name))]
            dependencies[mid] = set(outs)
        return dependencies

    def validate(self):
        "Validate the Dataflow, returning [] if it is valid or the invalid modules otherwise."
        invalid = []
        for module in self._modules.values():
            if not module.validate():
                logger.error('Cannot validate module %s', module.name)
                invalid.append(module)
        return invalid

    def __len__(self):
        return len(self._modules)

Dataflow.default = Dataflow()
