import datetime
import calendar

from typing import Any, Union, Tuple, Callable


def week_day_int(vec: Tuple[int, ...]) -> int:
    return datetime.datetime(*vec).weekday()  # type: ignore


def week_day(vec: Tuple[int, ...]) -> str:
    return calendar.day_name[week_day_int(vec)]


def ymd_string(vec: Tuple[int, ...]) -> str:
    y, m, d, *_ = vec
    return f"{y}-{m}-{d}"


def is_weekend(vec: Tuple[int, ...]) -> bool:
    return week_day_int(vec) >= 5


class _Unchanged:
    pass


UNCHANGED = _Unchanged()
UBool = Union[_Unchanged, bool]
Op = Callable[[Any, Any], Any]


def make_if_else(
    op_: Op, test_val: Any, if_true: UBool = UNCHANGED, if_false: UBool = UNCHANGED
) -> Callable[[Any], Any]:
    assert if_true != UNCHANGED or if_false != UNCHANGED

    def _fun(x: Any) -> Any:
        return if_true if op_(x, test_val) else if_false

    def _fun_if_true_unchanged(x: Any) -> Any:
        return x if op_(x, test_val) else if_false

    def _fun_if_false_unchanged(x: Any) -> Any:
        return if_true if op_(x, test_val) else x

    if if_true is UNCHANGED:
        return _fun_if_true_unchanged
    if if_false is UNCHANGED:
        return _fun_if_false_unchanged
    return _fun
