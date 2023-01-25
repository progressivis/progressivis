from .table import PDataExpr, Pipeable
from progressivis.stats import Min, Max

# pylint:disable=redefined-builtin

min = Pipeable(PDataExpr, Min)
max = Pipeable(PDataExpr, Max)
