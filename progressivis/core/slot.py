"""
Slots between modules.
"""
from __future__ import absolute_import, division, print_function

import logging
from collections import namedtuple

import six
from .utils import ProgressiveError
from .changemanager_base import EMPTY_BUFFER

logger = logging.getLogger(__name__)

class SlotDescriptor(namedtuple('SD',
                                ['name', 'type', 'required', 'doc'])):
    "SlotDescriptor is used in modules to describe the input/output slots."
    __slots__ = ()
    def __new__(cls, name, type=None, required=True, doc=None):
        # pylint: disable=redefined-builtin
        return super(SlotDescriptor, cls).__new__(cls,
                                                  name, type, required, doc)

@six.python_2_unicode_compatible
class Slot(object):
    "A Slot manages one connection between two modules."
    def __init__(self, output_module, output_name, input_module, input_name):
        self.output_name = output_name
        self.output_module = output_module
        self.input_name = input_name
        self.input_module = input_module
        self._name = None
        self.changes = None
        self._manage_columns = None
        self._last_columns = None

    def name(self):
        "Return the unique name of that slot"
        if self._name is None:
            self._name = self.input_module.id + '_' + self.input_name
        return self._name

    def data(self):
        "Return the data associated with this slot"
        return self.output_module.get_data(self.output_name)

    def scheduler(self):
        "Return the scheduler associated with this slot"
        return self.output_module.scheduler()

    @property
    def lock(self):
        "Return a context manager locking this slot for multi-threaded access"
        return self.input_module.lock

    def __str__(self):
        return six.u('%s(%s[%s]->%s[%s])' % (self.__class__.__name__,
                                             self.output_module.id,
                                             self.output_name,
                                             self.input_module.id,
                                             self.input_name))

    def __repr__(self):
        return str(self)

    def last_update(self):
        "Return the time of the last update for thie slot"
        if self.changes:
            return self.changes.last_update()
        return self.input_module.last_update()

    def to_json(self):
        "Return a dictionary describing this slot, meant to be serialized in json"
        return {'output_name': self.output_name,
                'output_module': self.output_module.id,
                'input_name': self.input_name,
                'input_module': self.input_module.id}

    def connect(self):
        "Run when the progressive pipeline is about to run through this slot"
        scheduler = self.scheduler()
        if scheduler != self.input_module.scheduler():
            raise ProgressiveError('Cannot connect modules managed by'
                                   ' different schedulers')

        # TODO we should ensure that all connections required to move the
        # pipeline from a valid state to another are executed atomically
        with scheduler.lock:
            # pylint: disable=protected-access
            scheduler.slots_updated()

            self.output_module._connect_output(self)
            prev_slot = self.input_module._connect_input(self)
            if prev_slot:
                raise ProgressiveError('Input already connected for %s',
                                       six.u(self))
            scheduler.invalidate()

    def validate_types(self):
        "Validate the types of the endpoints connected through this slot"
        output_type = self.output_module.output_slot_type(self.output_name)
        input_type = self.input_module.input_slot_type(self.input_name)
        if output_type is None or input_type is None:
            return True
        if issubclass(output_type, input_type):
            return True
        if (not isinstance(input_type, type) and callable(input_type) and
                input_type(output_type)):
            return True

        logger.error('Incompatible types for slot (%s,%s) in %s',
                     input_type, output_type, str(self))
        return False

    def create_changes(self,
                       buffer_created=True,
                       buffer_updated=False,
                       buffer_deleted=False):
        "Create a ChangeManager associated with the type of the endpoints of the slot"
        data = self.data()
        if data is not None:
            return self.create_changemanager(type(data), self,
                                             buffer_created=buffer_created,
                                             buffer_updated=buffer_updated,
                                             buffer_deleted=buffer_deleted)
        return None

    def update(self, run_number,
               buffer_created=True, buffer_updated=True, buffer_deleted=True,
               manage_columns=True):
        # pylint: disable=too-many-arguments
        "Compute the changes that occur since the last time this slot has been updated"
        if self.changes is None:
            self.changes = self.create_changes(buffer_created=buffer_created,
                                               buffer_updated=buffer_updated,
                                               buffer_deleted=buffer_deleted)
        if self._manage_columns is None:
            self._manage_columns = manage_columns
        if self.changes is None:
            return None
        with self.lock:
            df = self.data()
            return self.changes.update(run_number, df, self.name())

    @property
    def column_changes(self):
        "Compute the column changes since the last time this slot has been updated"
        if self._manage_columns:
            return self.changes.column_changes
        return None

    def reset(self):
        "Reset the slot"
        if self.changes:
            self.changes.reset(self.name())

    def clear_buffers(self):
        "Clear all the buffers"
        if self.changes:
            self.changes.clear()

    def has_buffered(self):
        """
        Return True if any of the created/updated/deleted information
        is buffered
        """
        return self.changes.has_buffered() if self.changes else False

    @property
    def created(self):
        "Return the buffer for created rows"
        return self.changes.created if self.changes else EMPTY_BUFFER

    @property
    def updated(self):
        "Return the buffer for updated rows"
        return self.changes.updated if self.changes else EMPTY_BUFFER

    @property
    def deleted(self):
        "Return the buffer for deleted rows"
        return self.changes.deleted if self.changes else EMPTY_BUFFER

    @property
    def changemanager(self):
        "Return the ChangeManager"
        return self.changes

    changemanager_classes = {}

    @staticmethod
    def create_changemanager(datatype, slot,
                             buffer_created,
                             buffer_updated,
                             buffer_deleted):
        """
        Create the ChangeManager responsible for this slot type or
        None if no ChangeManager is registered for that type.
        """
        # pylint: disable=too-many-arguments
        logger.debug('create_changemanager(%s, %s)', datatype, slot)
        if datatype is not None:
            queue = [datatype]
            processed = set()
            while queue:
                datatype = queue.pop()
                if datatype in processed:
                    continue
                processed.add(datatype)
                cls = Slot.changemanager_classes.get(datatype)
                if cls is not None:
                    logger.info('Creating changemanager %s for datatype %s'
                                ' of slot %s', cls, datatype, slot)
                    return cls(slot,
                               buffer_created,
                               buffer_updated,
                               buffer_deleted)
                if hasattr(datatype, '__base__'):
                    queue.append(datatype.__base__)
                elif hasattr(datatype, '__bases__'):
                    queue += datatype.__bases__
        logger.info('Creating no changemanager for datatype %s of slot %s',
                    datatype, slot)
        return None

    @staticmethod
    def add_changemanager_type(datatype, cls):
        """
        Declare a ChangerManager class for a slot type
        """
        assert isinstance(datatype, type)
        assert isinstance(cls, type)
        Slot.changemanager_classes[datatype] = cls


class InputSlots(object):
    # pylint: disable=too-few-public-methods
    """
    Convenience class to refer to input slots by name
    as if they were attributes.
    """
    def __init__(self, module):
        self.__dict__['module'] = module

    def __setattr__(self, name, slot):
        if not isinstance(slot, Slot):
            raise ProgressiveError('Assignment to input slot %s'
                                   ' should be a Slot', name)
        if slot.output_module is None or slot.output_name is None:
            raise ProgressiveError('Assignment to input slot %s invalid, '
                                   'missing slot output specs', name)
        slot.input_module = self.__dict__['module']
        slot.input_name = name
        slot.connect()

    def __getattr__(self, name):
        raise ProgressiveError('Input slots cannot be read, only assigned to')

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setitem__(self, name, slot):
        return self.__setattr__(name, slot)

    def __dir__(self):
        return self.__dict__['module'].input_slot_names()

    # TODO add a disconnect method to unregister the update manager


class OutputSlots(object):
    # pylint: disable=too-few-public-methods
    """
    Convenience class to refer to output slots by name
    as if they were attributes.
    """
    def __init__(self, module):
        self.__dict__['module'] = module

    def __setattr__(self, name, slot):
        raise ProgressiveError('Output slots cannot be assigned, only read')

    def __getattr__(self, name):
        return self.__dict__['module'].create_slot(name, None, None)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __dir__(self):
        return self.__dict__['module'].output_slot_names()
