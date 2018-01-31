from __future__ import absolute_import, division, print_function

from .table import TableExpr, Pipeable
from progressivis.stats import Min, Max

# pylint:disable=redefined-builtin

min = Pipeable(TableExpr, Min)
max = Pipeable(TableExpr, Max)
