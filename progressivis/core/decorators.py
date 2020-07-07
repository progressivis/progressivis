from functools import wraps, partial

class NoGo(Exception): pass

class _CtxImpl:
    def __init__(self):
        self._has_buffered = []

class _Context:
    def __init__(self):
        self._impl = _CtxImpl()
        self._parsed = False
        self._slot_policy = None
        self._slot_expr = []
        
    def reset(self):
        print("Reset ctx")
        self._impl = _CtxImpl()

    def __enter__(self):
        self._parsed = True
        #import pdb;pdb.set_trace()
        #raise NoGo()
        print("ENTER", self._slot_policy, self._slot_expr)
        return self._impl
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()



def process_slot(name, reset_if=True, exit_if=False, reset_cb=None):
    """
    this function includes reset_if, exit_if,  reset_cb in the closure
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
                else:
                    reset_cb(self)
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

_RULES = dict(run_if_all="or_if_all", run_if_any="and_if_any")
_INV_RULES = {v:k for (k, v) in _RULES.items()}
def accepted_first(s):
    return s in _RULES

def _slot_policy_rule(decname, *slots):
    """
    this function include *args in the closure
    """
    def decorator_(to_decorate):
        """
        this is the decorator.  it combines the decoration
        with the function to be decorated
        """
        @wraps(to_decorate)
        def decoration_(self, *args, **kwargs):
            """
            this function makes the decoration
            """
            #import pdb;pdb.set_trace()
            if self.context is None:
                    raise ValueError("context not found. consider processing slots before")
            if not self.context._parsed:
                print(f"Process {decname}")
                if self.context._slot_policy is None:
                    if not accepted_first(decname):
                        raise ValueError(f"{decname} must follow {_INV_RULES[decname]}")
                    self.context._slot_policy = decname
                elif decname != _RULES[self.context._slot_policy]: # first exists
                    raise ValueError(f"{decname} cannot follow {self.context._slot_policy}")
                elif not len(slots):
                    raise ValueError(f"run_if_all without arguments must be unique")
                self.context._slot_expr.append(slots)
            return to_decorate(self, *args, **kwargs) 
        return decoration_
    return decorator_


run_if_all = partial(_slot_policy_rule, "run_if_all")
or_if_all = partial(_slot_policy_rule, "or_if_all")
run_if_any = partial(_slot_policy_rule, "run_if_any")
and_if_any = partial(_slot_policy_rule, "and_if_any")


def check_slots(to_decorate):
    """
    """
    def simple_wrapper(self, run_number, step_size, howlong):
        """
        Cette fonction constitue
        la décoration
        """
        test = False
        self._parsed = True
        if test:
            return to_decorate(self, run_number, step_size, howlong)
        return self._return_run_step(self.state_blocked, steps_run=0)
    return simple_wrapper
