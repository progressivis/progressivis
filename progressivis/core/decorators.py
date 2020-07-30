from functools import wraps, partial

class _CtxImpl:
    def __init__(self):
        self._has_buffered = set()

class _Context:
    def __init__(self):
        self._impl = _CtxImpl()
        self._parsed = False
        self._checked = False
        self._slot_policy = None
        self._slot_expr = []
        
    def reset(self):
        self._impl = _CtxImpl()

    def __enter__(self):
        self._parsed = True
        if not self._checked:
            raise ValueError("mandatory @run_if_... decorator is missing!")
        return self._impl
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()



def process_slot(*names, reset_if=('update', 'delete'), exit_if=False, reset_cb=None):
    """
    this function includes reset_if, exit_if,  reset_cb in the closure
    """
    #import pdb;pdb.set_trace()
    if isinstance(reset_if, str):
        assert reset_if in ('update', 'delete')
        reset_if = (reset_if,)
    elif not reset_if:
        reset_if = tuple()
    else:
        assert set(reset_if) == set(('update', 'delete'))
    def run_step_decorator(run_step_):
        """
        run_step() decorator
        """
        #print("process slot deco", names)
        @wraps(run_step_)
        def run_step_wrapper(self, run_number, step_size, howlong):
            """
            decoration
            """
            #print("process slot wrapper", names, run_number)
            if self.context is None:
                self.context = _Context()
            for name in names:
                slot = self.get_input_slot(name)
                # slot.update(run_number)
                if exit_if and (slot.data() is None or not slot.has_buffered()):
                    return self._return_run_step(self.state_blocked, steps_run=0)            
                if ('update' in reset_if and slot.updated.any()) or\
                        ('delete' in reset_if and slot.deleted.any()):
                    slot.reset()
                    slot.update(run_number)
                    if isinstance(reset_cb, str):
                        getattr(self, reset_cb)()
                    else:
                        reset_cb(self)
                setattr(self.context._impl, name, slot)
                if slot.has_buffered():
                    self.context._impl._has_buffered.add(name)
            calc = run_step_(self, run_number, step_size, howlong) 
            # NB: "run_step_" fait partie de la fermeture
            return calc
        return run_step_wrapper
    return run_step_decorator

_RULES = dict(run_if_all="or_if_all", run_if_any="and_if_any", run_always="run_always")
_INV_RULES = {v:k for (k, v) in _RULES.items()}
def accepted_first(s):
    return s in _RULES

def _slot_policy_rule(decname, *slots_maybe):
    """
    this function includes *args in the closure
    """
    called_with_args = (not slots_maybe) or isinstance(slots_maybe[0], str)
    slots = slots_maybe if called_with_args else tuple([])
    assert called_with_args or callable(slots_maybe[0])
    def decorator_(to_decorate):
        """
        this is the decorator.  it combines the decoration
        with the function to be decorated
        """
        #print("policy deco", slots_maybe)
        has_hidden_attr = hasattr(to_decorate, "_hidden_progressivis_attr")
        @wraps(to_decorate)
        def decoration_(self, *args, **kwargs):
            """
            this function makes the decoration
            """
            #import pdb;pdb.set_trace()
            #print("policy wrapper", decname, slots_maybe, args, to_decorate.__name__, has_hidden_attr)     
            if self.context is None:
                    raise ValueError("context not found. consider processing slots before")
            if not self.context._parsed:
                if self.context._slot_policy is None:
                    if not accepted_first(decname):
                        raise ValueError(f"{decname} must follow {_INV_RULES[decname]}")
                    self.context._slot_policy = decname
                elif (self.context._slot_policy == "run_always" or
                      decname != _RULES[self.context._slot_policy]): # first exists and is not compatble
                    raise ValueError(f"{decname} cannot follow {self.context._slot_policy}")
                elif self.context._slot_expr == [tuple()]:
                    raise ValueError(f"{decname} without arguments must be unique")
                elif not accepted_first(decname) and not slots:
                    raise ValueError(f"{decname} requires arguments")
                self.context._slot_expr.append(slots)
            if not has_hidden_attr: # i.e. to_decorate is the genuine run_step
                self.context._parsed = True
                self.context._checked = True
                if not run_step_required(self):
                    return self._return_run_step(self.state_blocked, steps_run=0)
            return to_decorate(self, *args, **kwargs)
        decoration_._hidden_progressivis_attr = True
        return decoration_
    if called_with_args:
        return decorator_
    return decorator_(slots_maybe[0])

run_if_all = partial(_slot_policy_rule, "run_if_all")
or_all = partial(_slot_policy_rule, "or_if_all")
run_if_any = partial(_slot_policy_rule, "run_if_any")
and_any = partial(_slot_policy_rule, "and_if_any")
run_always = partial(_slot_policy_rule, "run_always")


def run_step_required(self_):
    policy = self_.context._slot_policy
    slot_expr = self_.context._slot_expr
    if slot_expr == [tuple()]:
        slot_expr = [[k for k in self_.input_descriptors.keys() if k!='_params']]
        self_.context._slot_expr = slot_expr
    if policy == "run_if_all": # i.e. all() or all() ...
        for grp in slot_expr:
            grp_test = True
            for elt in grp:
                if elt not in self_.context._impl._has_buffered:
                    grp_test = False
                    break
            if grp_test:
                return True
        return False
    elif policy == "run_if_any": # i.e. any() and any()
        for grp in slot_expr:
            grp_test = False
            for elt in grp:
                if elt in self_.context._impl._has_buffered:
                    grp_test = True
                    break
            if not grp_test:
                return False
        return True
    elif policy == "run_always":
        return True
    else:
        raise ValueError("Unknown slot policy")
