from progressive.core.scheduler import Scheduler, default_scheduler

class SchedulerServer(Scheduler):
    def __init__(self, fd):
        super(SchedulerServer, self).__init__()
        self.pipe = fd
        self._current_run = 0

    def _before_run(self, run_number):
        self._current_run = run_number

    def _after_run(self, run_number):
        self._current_run = run_number

    def request(self, req):
        print "Received request %s" % req.path
        
        if req.path == REQ_ECHO:
            return Response(req.serial, RES_ECHO, **req.params)
        elif req.path == REQ_WORFLOW:
            return Response(req.serial, RES_WORKFLOW, self.get_workflow())
        return response_error(req.serial, req)

    def get_workflow(self):
        nodes = []
        links = []
        for (mid,module) in self.modules().iteritems():
            node = {'id': mid, 'type': module.__class__.__name__}
            nodes.append(node)
            for slot in module.input_slot_values():
                if slot is None:
                    continue
                l = {'source': slot.output_module.id(), 'target': mid,
                     'source_slot': slot.output_name, 'target_slot': slot.input_name}
                links.add(l)
        return {'nodes': nodes, 'links': links}

    def serve_once(self):
        req = self.pipe.recv()
        res = self.request(req)
        self.pipe.send(res)
        return res.path != RES_TERMINATE

def run_scheduler_server(pipe):
    global default_scheduler
    default_scheduler = SchedulerServer(pipe)
    while default_scheduler.serve_once():
        pass

