"""
Slots between modules.
"""

import logging
from collections import namedtuple
from .changemanager_base import EMPTY_BUFFER
from .changemanager_literal import LiteralChangeManager

logger = logging.getLogger(__name__)

class SlotDescriptor(namedtuple('SD',
                                ['name', 'type', 'required', 'multiple', 'datashape',
                                 'buffer_created',
                                 'buffer_updated',
                                 'buffer_deleted',
                                 'buffer_exposed',
                                 'buffer_masked'])):
    "SlotDescriptor is used in modules to describe the input/output slots."
    __slots__ = ()

    def __new__(cls, name, type=None, required=True, multiple=False, datashape=None,
                buffer_created=True,
                buffer_updated=True,
                buffer_deleted=True,
                buffer_exposed=True,
                buffer_masked=True):
        # pylint: disable=redefined-builtin
        return super(SlotDescriptor, cls).__new__(cls, name, type,
                                                  required, multiple, datashape,
                                                  buffer_created,
                                                  buffer_updated,
                                                  buffer_deleted,
                                                  buffer_exposed,
                                                  buffer_masked)


class Slot(object):
    "A Slot manages one connection between two modules."
    def __init__(self, output_module, output_name, input_module, input_name):
        self.output_name = output_name
        self.output_module = output_module
        self.input_name = input_name
        self.input_module = input_module
        self.original_name = None
        self._name = None
        self.changes = None

    def name(self):
        "Return the name of the slot"
        if self._name is None:
            self._name = (self.input_module.name + '_' + self.input_name)
        return self._name

    def data(self):
        "Return the data associated with this slot"
        return self.output_module.get_data(self.output_name)

    def scheduler(self):
        "Return the scheduler associated with this slot"
        return self.output_module.scheduler()

    def input_descriptor(self):
        if self.original_name:
            return self.input_module.input_slot_descriptor(self.original_name)
        return self.input_module.input_slot_descriptor(self.input_name)

    def output_descriptor(self):
        return self.output_module.output_slot_descriptor(self.output_name)

    def __str__(self):
        return '%s(%s[%s]->%s[%s])' % (self.__class__.__name__,
                                       self.output_module.name,
                                       self.output_name,
                                       self.input_module.name,
                                       self.input_name)

    def __repr__(self):
        return str(self)

    def last_update(self):
        "Return the time of the last update for thie slot"
        if self.changes:
            return self.changes.last_update()
        return self.input_module.last_update()

    def to_json(self):
        """
        Return a dictionary describing this slot, meant to be
        serialized in json.
        """
        return {'output_name': self.output_name,
                'output_module': self.output_module.name,
                'input_name': self.input_name,
                'input_module': self.input_module.name}

    def connect(self):
        "Declares the connection in the Dataflow"
        dataflow = self.output_module.dataflow()  # also in input_module?
        dataflow.add_connection(self)

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
                       buffer_deleted=False,
                       buffer_exposed=False,
                       buffer_masked=False):
        "Create a ChangeManager associated with the type of the slot's data."
        data = self.data()
        if data is not None:
            return self.create_changemanager(type(data), self,
                                             buffer_created=buffer_created,
                                             buffer_updated=buffer_updated,
                                             buffer_deleted=buffer_deleted,
                                             buffer_exposed=buffer_exposed,
                                             buffer_masked=buffer_masked)
        return None

    def update(self, run_number,
               buffer_created=True, buffer_updated=True, buffer_deleted=True,
               buffer_exposed=True, buffer_masked=True, manage_columns=True):
        # pylint: disable=too-many-arguments
        "Compute the changes that occur since this slot has been updated."
        if self.changes is None:
            desc = self.input_descriptor()
            # create_changes always return a ChangeManager
            self.changes = self.create_changes(
                buffer_created=desc.buffer_created,
                buffer_updated=desc.buffer_updated,
                buffer_deleted=desc.buffer_deleted,
                buffer_exposed=desc.buffer_exposed,
                buffer_masked=desc.buffer_masked)
        if self.changes:
            df = self.data()
            self.changes.update(run_number, df, self.name())

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
    def perm_deleted(self):
        "Return the buffer for deleted rows"
        return self.changes.perm_deleted if self.changes else EMPTY_BUFFER

    @property
    def masked(self):
        "Return the buffer for deleted rows"
        return self.changes.masked if self.changes else EMPTY_BUFFER

    @property
    def deleted_masked(self):
        "Return the buffer for deleted rows"
        return self.changes.deleted_masked if self.changes else EMPTY_BUFFER


    @property
    def changemanager(self):
        "Return the ChangeManager"
        return self.changes

    changemanager_classes = {}

    @staticmethod
    def create_changemanager(datatype, slot,
                             buffer_created,
                             buffer_updated,
                             buffer_deleted,
                             buffer_exposed,
                             buffer_masked):
        """
        Create the ChangeManager responsible for this slot type or
        None if no ChangeManager is registered for that type.
        """
        # pylint: disable=too-many-arguments
        logger.debug('create_changemanager(%s, %s)', datatype, slot)
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
                           buffer_deleted,
                           buffer_exposed,
                           buffer_masked)
            if hasattr(datatype, '__base__'):
                queue.append(datatype.__base__)
            elif hasattr(datatype, '__bases__'):
                queue += datatype.__bases__
        logger.info('Creating LiteralChangeManager for datatype %s of slot %s',
                    datatype, slot)
        return None
        return LiteralChangeManager(slot,
                                    buffer_created,
                                    buffer_updated,
                                    buffer_deleted,
                                    buffer_exposed,
                                    buffer_masked)

    @staticmethod
    def add_changemanager_type(datatype, cls):
        """
        Declare a ChangerManager class for a slot type
        """
        assert isinstance(datatype, type)
        assert isinstance(cls, type)
        Slot.changemanager_classes[datatype] = cls
