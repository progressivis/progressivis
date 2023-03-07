from __future__ import annotations

from functools import wraps, partial

from typing import (
    Callable,
    Sequence,
    Tuple,
    Union,
    Any,
    TYPE_CHECKING,
    Optional,
    List,
    Set,
)

if TYPE_CHECKING:
    from .module import Module, ReturnRunStep
    from .slot import Slot

    RunStepCallable = Callable[[Any, int, int, float], ReturnRunStep]
    DecCallable = Callable[[RunStepCallable], RunStepCallable]


class _CtxImpl:
    def __init__(self) -> None:
        self._has_buffered: Set[str] = set()

    def __getattr__(self, name: str) -> Slot:
        return super().__getattr__(self, name)  # type: ignore


class _Context:
    def __init__(self) -> None:
        self._impl = _CtxImpl()
        self._parsed: bool = False
        self._checked: bool = False
        self._slot_policy: Optional[str] = None
        self._slot_expr: List[Union[Sequence[Any], str]] = []

    def reset(self) -> None:
        self._impl = _CtxImpl()

    def __enter__(self) -> _CtxImpl:
        self._parsed = True
        if not self._checked:
            raise ValueError("mandatory @run_if_... decorator is missing!")
        return self._impl

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.reset()


def process_slot(
    *names: str,
    reset_if: Union[bool, str, Tuple[str, ...]] = ("update", "delete"),
    reset_cb: Union[None, str, Callable[[Module], None]] = None,
) -> Callable[[RunStepCallable], RunStepCallable]:
    """
    this function includes reset_if, reset_cb in the closure
    """
    if isinstance(reset_if, str):
        assert reset_if in ("update", "delete")
        reset_if = (reset_if,)
    elif not reset_if:
        reset_if = tuple()
    else:
        assert reset_if == ("update", "delete") or reset_if == ("delete", "update")
    assert isinstance(reset_if, tuple)

    def run_step_decorator(run_step_: RunStepCallable) -> RunStepCallable:
        """
        run_step() decorator
        """
        @wraps(run_step_)
        def run_step_wrapper(
            self: Module, run_number: int, step_size: int, howlong: float
        ) -> ReturnRunStep:
            """
            decoration
            """
            if self.context is None:
                self.context = _Context()
            reset_all = False
            # check if at least one slot fill the reset condition
            for name in names:
                slot = self.get_input_slot(name)
                # slot.update(run_number)
                assert isinstance(reset_if, tuple)
                if ("update" in reset_if and slot.updated.any()) or (
                    "delete" in reset_if and slot.deleted.any()
                ):
                    reset_all = True
                    break
            # if True (reset_all) thel all slots are reseted
            if reset_all:
                for name in names:
                    slot = self.get_input_slot(name)
                    # slot.update(run_number)
                    slot.reset()
                    slot.update(run_number)
                if isinstance(reset_cb, str):
                    getattr(self, reset_cb)()
                elif reset_cb is not None:
                    reset_cb(self)
            # all slots are added to the context
            for name in names:
                slot = self.get_input_slot(name)
                setattr(self.context._impl, name, slot)
                if slot.has_buffered():
                    self.context._impl._has_buffered.add(name)
            return run_step_(self, run_number, step_size, howlong)

        return run_step_wrapper

    return run_step_decorator


_RULES = dict(run_if_all="or_if_all", run_if_any="and_if_any", run_always="run_always")
_INV_RULES = {v: k for (k, v) in _RULES.items()}


def accepted_first(s: str) -> bool:
    return s in _RULES


def _slot_policy_rule(decname: str, *slots_maybe: str) -> RunStepCallable:
    """
    this function includes *args in the closure
    """
    called_with_args = len(slots_maybe) == 0 or isinstance(slots_maybe[0], str)
    slots: Sequence[str] = slots_maybe if called_with_args else tuple([])
    assert called_with_args or callable(slots_maybe[0])

    def decorator_(to_decorate: RunStepCallable) -> RunStepCallable:
        """
        this is the decorator.  it combines the decoration
        with the function to be decorated
        """
        has_hidden_attr = hasattr(to_decorate, "_hidden_progressivis_attr")

        @wraps(to_decorate)
        def decoration_(
            self: Module, run_number: int, step_size: int, howlong: float
        ) -> ReturnRunStep:
            """
            this function makes the decoration
            """
            if self.context is None:
                raise ValueError("context not found. consider processing slots before")
            if not self.context._parsed:
                if self.context._slot_policy is None:
                    if not accepted_first(decname):
                        raise ValueError(f"{decname} must follow {_INV_RULES[decname]}")
                    self.context._slot_policy = decname
                elif (
                    self.context._slot_policy == "run_always"
                    or decname != _RULES[self.context._slot_policy]
                ):  # first exists and is not compatble
                    raise ValueError(
                        f"{decname} cannot follow {self.context._slot_policy}"
                    )
                elif self.context._slot_expr == [tuple()]:
                    raise ValueError(f"{decname} without arguments must be unique")
                elif not accepted_first(decname) and not slots:
                    raise ValueError(f"{decname} requires arguments")
                self.context._slot_expr.append(slots)
            if not has_hidden_attr:  # i.e. to_decorate is the genuine run_step
                self.context._parsed = True
                self.context._checked = True
                if not run_step_required(self):
                    return self._return_run_step(self.state_blocked, steps_run=0)
            return to_decorate(self, run_number, step_size, howlong)

        decoration_._hidden_progressivis_attr = True  # type: ignore
        return decoration_

    if called_with_args:
        return decorator_  # type: ignore
    return decorator_(slots_maybe[0])  # type: ignore


run_if_all: DecCallable = partial(_slot_policy_rule, "run_if_all")
or_all: DecCallable = partial(_slot_policy_rule, "or_if_all")
run_if_any: DecCallable = partial(_slot_policy_rule, "run_if_any")
and_any: DecCallable = partial(_slot_policy_rule, "and_if_any")
run_always: DecCallable = partial(_slot_policy_rule, "run_always")


def run_step_required(self_: Module) -> bool:
    assert self_.context
    policy = self_.context._slot_policy
    slot_expr = self_.context._slot_expr
    if slot_expr == [tuple()]:
        slot_expr = [[k for k in self_.input_descriptors.keys() if k != "_params"]]
        self_.context._slot_expr = slot_expr
    if policy == "run_if_all":  # i.e. all() or all() ...
        for grp in slot_expr:
            grp_test = True
            for elt in grp:
                if elt not in self_.context._impl._has_buffered:
                    grp_test = False
                    break
            if grp_test:
                return True
        return False
    elif policy == "run_if_any":  # i.e. any() and any()
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
