from progressivis.core.common import ProgressiveError

import logging
logger = logging.getLogger(__name__)

class SlotDescriptor(object):
    def __init__(self, name, type=None, required=True, doc=None):
        self.name = name
        self.type = type
        self.required = required
        self.doc = doc

class Slot(object):
    def __init__(self, output_module, output_name, input_module, input_name):
        self.output_name = output_name
        self.output_module = output_module
        self.input_name = input_name
        self.input_module = input_module

    def data(self):
        return self.output_module.get_data(self.output_name)

    @property
    def lock(self):
        return self.input_module.lock

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return u'%s(%s[%s]->%s[%s])' % (self.__class__.__name__,
                                        self.output_module.id,
                                         self.output_name,
                                         self.input_module.id,
                                         self.input_name)
    
    def __repr__(self):
        return self.__unicode__()

    def to_json(self):
        return {'output_name': self.output_name,
                'output_module': self.output_module.id,
                'input_name': self.input_name,
                'input_module': self.input_module.id}
    
    def connect(self):
        scheduler = self.output_module.scheduler()
        if scheduler != self.input_module.scheduler():
            raise ProgressiveError('Cannot connect modules managed by different schedulers')
        
        # TODO we should ensure that all connections required to move the pipeline from a valid state to another are executed atomically
        with scheduler.lock:
            scheduler.slots_updated()

            self.output_module._connect_output(self)
            prev_slot = self.input_module._connect_input(self)
            if prev_slot:
                raise ProgressiveError(u'Input already connected for %s', unicode(self))
            scheduler.invalidate()

    def validate_types(self):
        output_type = self.output_module.output_slot_type(self.output_name)
        input_type = self.input_module.input_slot_type(self.input_name)
        if output_type is None or input_type is None:
            return True
        if output_type == input_type:
            return True
        logger.error('Incompatible types for slot (%s,%s) in %s', input_type, output_type, str(self))
        return False #TODO: more compatibility comes here

class InputSlots(object):
    def __init__(self, module):
        self.__dict__['module'] = module

    def __setattr__(self, name, slot):
        if not isinstance(slot, Slot):
            raise ProgressiveError('Assignment to input slot %s should be a Slot', name)
        if slot.output_module is None or slot.output_name is None:
            raise ProgressiveError('Assignment to input slot %s invalid, missing slot output specs', name)
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
