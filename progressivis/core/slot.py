from __future__ import absolute_import, division, print_function

import six
import logging
from .utils import ProgressiveError
from .changemanager_base import EMPTY_BUFFER

logger = logging.getLogger(__name__)

builtin_type = type


class SlotDescriptor(object):
    # pylint: disable=redefined-builtin
    def __init__(self, name, type=None, required=True, doc=None):
        assert type is None or isinstance(type, builtin_type)
        self.name = name
        self.type = type
        self.required = required
        self.doc = doc


@six.python_2_unicode_compatible
class Slot(object):
    def __init__(self, output_module, output_name, input_module, input_name):
        self.output_name = output_name
        self.output_module = output_module
        self.input_name = input_name
        self.input_module = input_module
        self.changes = None

    def data(self):
        return self.output_module.get_data(self.output_name)

    def scheduler(self):
        return self.output_module.scheduler()

    @property
    def lock(self):
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
        if self.changes:
            return self.changes.last_update()
        return self.input_module.last_update()

    def to_json(self):
        return {'output_name': self.output_name,
                'output_module': self.output_module.id,
                'input_name': self.input_name,
                'input_module': self.input_module.id}

    def connect(self):
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
        output_type = self.output_module.output_slot_type(self.output_name)
        input_type = self.input_module.input_slot_type(self.input_name)
        if output_type is None or input_type is None:
            return True
        if output_type == input_type:
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
                       manage_columns=True):
        data = self.data()
        if data is not None:
            return self.create_changemanager(type(data), self,
                                             buffer_created=buffer_created,
                                             buffer_updated=buffer_updated,
                                             buffer_deleted=buffer_deleted,
                                             manage_columns=manage_columns)
        return None

    def update(self, run_number, mid,
               buffer_created=True, buffer_updated=True, buffer_deleted=True,
               manage_columns=True):
        if self.changes is None:
            self.changes = self.create_changes(buffer_created=buffer_created,
                                               buffer_updated=buffer_updated,
                                               buffer_deleted=buffer_deleted,
                                               manage_columns=manage_columns)
        if self.changes is None:
            return
        with self.lock:
            df = self.data()
            return self.changes.update(run_number, df, mid=mid)

    def reset(self, mid=None):
        if self.changes:
            self.changes.reset(mid)

    def clear_buffers(self):
        if self.changes:
            self.changes.clear()

    def has_buffered(self):
        return self.changes.has_buffered() if self.changes else False

    @property
    def created(self):
        return self.changes.created if self.changes else EMPTY_BUFFER

    @property
    def updated(self):
        return self.changes.updated if self.changes else EMPTY_BUFFER

    @property
    def deleted(self):
        return self.changes.deleted if self.changes else EMPTY_BUFFER

    @property
    def changemanager(self):
        return self.changes

    changemanager_classes = {}

    @staticmethod
    def create_changemanager(datatype, slot,
                             buffer_created,
                             buffer_updated,
                             buffer_deleted,
                             manage_columns):
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
                               buffer_deleted,
                               manage_columns)
                if hasattr(datatype, '__base__'):
                    queue.append(datatype.__base__)
                elif hasattr(datatype, '__bases__'):
                    queue += datatype.__bases__
        logger.info('Creating no changemanager for datatype %s of slot %s',
                    datatype, slot)
        return None

    @staticmethod
    def add_changemanager_type(datatype, cls):
        assert isinstance(datatype, type)
        assert isinstance(cls, type)
        Slot.changemanager_classes[datatype] = cls


class InputSlots(object):
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
