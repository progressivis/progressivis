from .table import PTableExpr, Pipeable
from progressivis.stats import Min, Max

# pylint:disable=redefined-builtin

min = Pipeable(PTableExpr, Min)
max = Pipeable(PTableExpr, Max)
