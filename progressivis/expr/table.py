from __future__ import absolute_import, division, print_function

from .core import Expr
from ..table.constant import Constant
from ..table.table import Table
from ..table.module import TableModule
from abc import ABCMeta, abstractmethod, abstractproperty


class TableExpr(Expr):
    def __init__(self,
                 module_class,
                 args,
                 kwds,
                 module=None,
                 output_slot=None,
                 output_slot_table="table"):
        super(TableExpr, self).__init__(
            module_class, args, kwds, module=module, output_slot=output_slot)
        self._output_slot_table = output_slot_table

    @property
    def table(self):
        return self._module.table()

    def select(self, columns):
        return TableExpr(Constant, self.table.loc[:, columns])

    # def filter(self, predicate):
    #     return TableExpr(Filter, self.module, predicate)


class Pipeable(object):
    def __init__(self,
                 expr_class,
                 module_class,
                 args=(),
                 kwds={},
                 repipe=None,
                 out=None):
        self._expr_class = expr_class
        self._module_class = module_class
        self._args = args
        self._kwds = kwds
        self._repipe = repipe
        self._repipe_out = out

    def __call__(self, *args, **kwds):
        _kw = dict(**kwds)
        rp_ = _kw.pop('repipe', None)
        if len(args) > 0:
            ret = self._expr_class(self._module_class, args, _kw)
            if rp_:
                return ret.repipe(rp_)
            else:
                return ret
            return ret
        return Pipeable(self._expr_class, self._module_class, args, _kw, rp_)

    def __or__(self, other):
        expr = self._expr_class(self._module_class, self._args, self._kwds)
        return expr | other

    def tee(self, lambda1, lambda2):
        lambda1(self)
        return lambda2(self)

    def repipe(self, mod_name, out=None):
        self._repipe = mod_name
        self._repipe_out = out
        return self


class PipedInput(object):
    def __init__(self, obj):
        self._obj = obj

    def __or__(self, other):
        return other._expr_class(other._module_class,
                                 (self._obj, ) + other._args, other._kwds)
