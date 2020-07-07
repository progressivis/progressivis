from functools import wraps


class _CtxImpl:
    def __init__(self):
        self._has_buffered = []
        print("HB:", self._has_buffered)

class _Context:
    def __init__(self):
        self._impl = _CtxImpl()
    def reset(self):
        print("Reset ctx")
        self._impl = _CtxImpl()
    def __enter__(self):
        return self._impl
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()



def process_slot(name, reset_if=True, exit_if=False, reset_cb=None):
    """
    this function incudes reset_if, exit_if in the closure
    """
    def run_step_decorator(run_step_):
        """
        run_step() decorator
        """
        @wraps(run_step_)
        def run_step_wrapper(self, run_number, step_size, howlong):
            """
            decoration
            """
            #if not "create_context_magic" in kwargs:
            #    raise ValueError("context not found")
            print("process slot", name)
            if self.context is None:
                self.context = _Context()
            slot = self.get_input_slot(name)
            slot.update(run_number)
            if exit_if and not slot.has_buffered():
                return self._return_run_step(self.state_blocked, steps_run=0)            
            if reset_if and (slot.updated.any() or slot.deleted.any()):
                slot.reset()
                slot.update(run_number)
                if isinstance(reset_cb, str):
                    getattr(self, reset_cb)(self)
            setattr(self.context._impl, name, slot)
            if slot.has_buffered():
                #import pdb;pdb.set_trace()
                self.context._impl._has_buffered.append(name)
            calc = run_step_(self, run_number, step_size, howlong) 
            # NB: "run_step_" fait partie de la fermeture
            return calc
        return run_step_wrapper
    return run_step_decorator

def slot_expr(expr):
    def run_step_decorator(run_step_):
        """
        run_step() decorator
        """
        @wraps(run_step_)
        def run_step_wrapper(self, run_number, step_size, howlong):
            """
            """
            if expr == "ANY":
                for k in self.input_descriptors.keys():
                    if k in self.context._impl._has_buffered:
                        print("buffered", k)
                        break
                else: # for
                    print("NOT ANY")
                    return self._return_run_step(self.state_blocked, steps_run=0)
            elif expr == "ALL":
                for k in self.input_descriptors.keys():
                    if k not in self.context._impl._has_buffered:
                        return self._return_run_step(self.state_blocked, steps_run=0)
            else:
                raise ValueError("Not yet implemented")
            calc = run_step_(self, run_number, step_size, howlong)
            # NB: "run_step_" fait partie de la fermeture
            return calc
        return run_step_wrapper
    return run_step_decorator
    
