from abc import ABCMeta, abstractmethod, abstractproperty
import six

@six.python_2_unicode_compatible
class ModuleImpl(six.with_metaclass(ABCMeta, object)):
    def __init__(self):
        self.is_started = False

    def __str__(self):
        return six.u('ModuleImpl %s: %s' % (self.__class__.__name__, id(self)))

    @abstractmethod
    def start(self, *args, **kwargs):
        pass

    @abstractmethod
    def resume(self, *args, **kwargs):
        pass    
