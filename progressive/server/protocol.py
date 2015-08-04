
class Request(object):
    def __init__(self, path, serial=None, value=None):
        self.serial = serial
        self.path = path
        self.value = value

class Response(object):
    def __init__(self, serial, path, done, value):
        self.serial = serial
        self.path = path
        self.done = done
        self.value = value

class Messages(object):
    ERROR = "/error"
    ECHO = "/echo"
    WORKFLOW_LOAD = "/api/workflow/load"
    WORKFLOW_STATUS = "/api/workflow/status"
    WORKFLOW_NETWORK = "/api/workflow/network"
    WORKFLOW_RUN = "/api/workflow/run"
    WORKFLOW_STOP = "/api/workflow/stop"

    ALL = [ ECHO, WORKFLOW_LOAD, WORKFLOW_STATUS, WORKFLOW_NETWORK, WORKFLOW_RUN, WORKFLOW_STOP ]

