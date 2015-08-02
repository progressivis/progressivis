
class Request(object):
    def __init__(self, path, serial=None, **params):
        self.serial = serial
        self.path = path
        self.params = params

class Response(object):
    def __init__(self, serial, path, done, value):
        self.serial = serial
        self.path = path
        self.done = done
        self.value = value

class Messages(object):
    ERROR = "/error"
    ECHO = "/echo"
    WORKFLOW_LOAD = "/workflow/load"
    WORKFLOW_STATUS = "/workflow/status"
    WORKFLOW_RUN = "/workflow/run"
    WORKFLOW_STOP = "/workflow/stop"

    ALL = [ ECHO, WORKFLOW_LOAD, WORKFLOW_STATUS, WORKFLOW_RUN, WORKFLOW_STOP ]
