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
    """
    Dataflow graph maintaining modules connected with slots.
    """
    def __init__(self, scheduler=None):
        if scheduler is None:
            scheduler = BaseScheduler.default
        if scheduler is None:
            raise ValueError('No scheduler specified and no default scheduler')
        self.scheduler = scheduler
        self._modules = scheduler.modules.copy()
        self._modules_created = None
        self._modules_deleted = None
        self._slots_created = None
        self._slots_deleted = None

    default = None

    def __enter__(self):
        assert self.default is None, "cannot have more than one default dataflow"
        Dataflow.default = self
        return self

    def __exit__(self, *args):
        assert self.default is self
        Dataflow.default = None

    @staticmethod
    def get_active_dataflow(dataflow=None):
        """
        Obtain the currently active dataflow instance by returning the explicitly given
        dataflow or using the default dataflow.

        Parameters
        ----------
        dataflow : Dataflow or None
            Dataflow to return or `None` to use the default dataflow.

        Raises
        ------
        ValueError
            If no `Dataflow` instance can be obtained.
        """
        dataflow = dataflow or Dataflow.default
        if not dataflow:
            raise ValueError("`dataflow` must be given explicitly"
                             " or a default dataflow must be set")
        return dataflow

    def __getitem__(self, name):
        return self._modules[name]

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

    def __contains__(self, moduleid):
        "Return True if the moduleid exists in this Dataflow."
        return moduleid in self._modules

    def add(self, module):
        "Add a module to this Dataflow."
        if not module.is_created():
            raise ProgressiveError('Cannot add running module %s' % module.name)
        if module.name is None:
            # pylint: disable=protected-access
            module._id = self.generate_id(module.pretty_typename())
        self._add_module(module)

    def _add_module(self, module):
        self._modules_created.append(module.name)
        #self._modules[module.name] = module

    def generate_id(self, prefix):
        "Generate an id for a module."
        for i in range(1, 10):
            mid = '%s_%d' % (prefix, i)
            if mid not in self._modules:
                return mid
        return '%s_%s' % (prefix, uuid4())
