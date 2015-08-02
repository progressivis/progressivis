from progressive.core.scheduler import Scheduler, default_scheduler
from progressive.server.protocol import Request, Response, Messages

class SchedulerServer(Scheduler):
    def __init__(self, fd):
        print "Creating SchedulerServer with pipe"
        super(SchedulerServer, self).__init__()
        self.pipe = fd
        self._current_run = 0

    def _before_run(self, run_number):
        self._current_run = run_number

    def _after_run(self, run_number):
        self._current_run = run_number

    def request(self, req):
        print "Received request %s" % req.path
        
        if req.path == Messages.ECHO:
            return Response(req.serial, req.path, True, req.params)
        elif req.path == Messages.WORKFLOW_LOAD:
            (ok, msg) = self.load(req.value)
            return Response(req.serial, req.path, ok, msg) 
        elif req.path == Messages.WORKFLOW_STATUS:
            return Response(req.serial, req.path, True, self.get_workflow())
        elif req.path == Messages.WORKFLOW_RUN:
            if len(self.modules()) == 0:
                return Response(req.serial, req.path, False, "Nothing to run")
            else:
                self.run()
                return Response(req.serial, req.path, True, "Running")
        elif req.path == Messages.WORKFLOW_STOP:
            self.stop()
            return Response(req.serial, req.path, True, "Stopped")
        return Response(req.serial, Messages.ERROR, False, "Unhandled request %s"%req.path)

    def get_workflow(self):
        nodes = []
        links = []
        for (mid,module) in self.modules().iteritems():
            node = {'id': mid, 'type': module.__class__.__name__, 'state': module.state }
            nodes.append(node)
            for slot in module.input_slot_values():
                if slot is None:
                    continue
                l = {'source': slot.output_module.id(), 'target': mid,
                     'source_slot': slot.output_name, 'target_slot': slot.input_name}
                links.add(l)
        return {'nodes': nodes, 'links': links}

    def load(self, workflow):
        try:
            execfile(workflow)
        except Exception as e:
            return (False, e.message)
        return (True, "Done")

    def serve_once(self):
        print "Ready to receive in SchedulerServer"
        req = self.pipe.recv()
        print "Received request %s(%d) in SchedulerServer"%(req.path,req.serial)
        res = self.request(req)
        self.pipe.send(res)
        return res.path != Messages.WORKFLOW_STOP

def run_scheduler_server(pipe):
    global default_scheduler
    default_scheduler = SchedulerServer(pipe)
    while default_scheduler.serve_once():
        pass


