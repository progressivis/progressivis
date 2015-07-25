from progressive.common import ProgressiveError

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

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return u'%s(%s[%s]->%s[%s])' % (self.__class__.__name__,
                                        self.output_module.id(),
                                         self.output_name,
                                         self.input_module.id(),
                                         self.input_name)
    
    def __repr__(self):
        return self.__unicode__()
    
    def connect(self):
        scheduler = self.output_module.scheduler()
        if scheduler != self.input_module.scheduler():
            raise ProgressiveError('Cannot connect modules managed by different schedulers')
        
        if scheduler.is_running():
            raise ProgressiveError('Cannot change module slots while running')

        self.output_module._connect_output(self)
        prev_slot = self.input_module._connect_input(self)
        if prev_slot:
            raise ProgressiveError(u'Input already connected for %s', unicode(self))
        scheduler.invalidate()

    def validate_types(self):
        output_type = self.output_module.get_output_type(self.output_name)
        input_type = self.input_module.get_input_type(self.input_name)
        if output_type is None or input_type is None:
            return True
        if output_type == input_type:
            return True
        #TODO: more compatibility comes here
        return False
