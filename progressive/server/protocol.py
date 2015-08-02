
class Request(object):
    def __init__(self, path, serial=None, **params):
        self.serial = serial
        self.path = path
        self.params = params

class Response(object):
    def __init__(self, serial, path, value):
        self.serial = serial
        self.path = path
        self.value = value

class Messages(object):
    ERROR = "/error"
    WORKFLOW = "/scheduler/workflow"
    ECHO = "/echo"
    TERMINATE = "/terminate"

    ALL = [ ECHO, WORKFLOW, TERMINATE ]
