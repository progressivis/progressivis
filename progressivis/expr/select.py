from __future__ import absolute_import, division, print_function

from ..table.constant import Constant
from .table import TableExpr


def select(self, columns):
    return TableExpr(Constant, self.table.loc[:,columns])
