
from ..table.constant import Constant
from .table import TableExpr


def select(self, columns):
    return TableExpr(Constant, self.table.loc[:,columns])
