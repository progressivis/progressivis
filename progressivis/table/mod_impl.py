from abc import ABCMeta, abstractmethod


class ModuleImpl(metaclass=ABCMeta):
    def __init__(self):
        self.is_started = False

    def __str__(self):
        return "ModuleImpl %s: %s" % (self.__class__.__name__, id(self))

    @abstractmethod
    def start(self, *args, **kwargs):
        pass

    @abstractmethod
    def resume(self, *args, **kwargs):
        pass
