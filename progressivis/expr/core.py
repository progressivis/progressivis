from __future__ import absolute_import, division, print_function

import logging
logger = logging.getLogger(__name__)
from ..table.module import TableModule


class ValidationError(RuntimeError):
    pass


def filter_underscore(lst):
    return [l for l in lst if not l.startswith('_')]


class Expr(object):
    def __init__(self, module_class, args, kwds, output_slot=None,
                 module=None):
        self._module_class = module_class
        lazy = kwds.pop('lazy', False)
        self._args = args
        self._kwds = kwds
        self._module = module
        self._output_slot = output_slot
        self._valid = (module is not None) or None
        self._expr_args = ()
        self._non_expr_args = ()
        self._expr_kwds = {}
        self._non_expr_kwds = {}
        self._repipe = None
        if not lazy:
            self.validate()

    @property
    def module(self):
        return self._module

    @property
    def output_slot(self):
        return self._output_slot

    def get_data(self, name):
        return self.module.get_data(name)

    def __getitem__(self, output_slot):
        self._module.get_output_slot(
            output_slot)  # raise an error if output_slot does not exist
        return Expr(
            self._module_class,
            self._non_expr_args,
            dict(**self._non_expr_kwds, lazy=True),
            output_slot=output_slot,
            module=self._module)

    def tee(self, lambda1, lambda2):
        lambda1(self)
        return lambda2(self)

    def _validate_args(self):
        modules = []
        non_modules = []
        for a in self._args:
            if isinstance(a, Expr):
                a.validate()
                modules.append(a)
            else:
                non_modules.append(a)
        self._expr_args = modules
        self._non_expr_args = non_modules

    def _validate_kwds(self):
        modules = {}
        non_modules = {}
        for (k, a) in self._kwds.items():
            if isinstance(a, Expr):
                a.validate()
                modules[k] = a
            else:
                non_modules[k] = a
        self._expr_kwds = modules
        self._non_expr_kwds = non_modules

    def _connect(self, module, expr, input_slot):
        input_module = expr.module
        output_slot = expr.output_slot
        if output_slot is None:
            output_slots = filter_underscore(input_module.output_slot_names())
            if len(output_slots) == 0:
                raise ValueError('Cannot extract output slot from module %s',
                                 input_module)
            output_slot = output_slots[0]  # take the first one
        if input_slot is None:
            input_slots = filter_underscore(module.input_slot_names())
            for inp in input_slots:
                slot = module.get_input_slot(inp)
                if slot is None:  # no input slot connected yet
                    input_slot = inp
                    break
            if input_slot is None:
                raise ValueError('Cannot extract input slot from module %s',
                                 module)
        input_module.connect_output(output_slot, module, input_slot)

    def _instanciate_module(self):
        module = self._module_class(*self._non_expr_args,
                                    **self._non_expr_kwds)
        for expr in self._expr_args:
            self._connect(module, expr, None)

        for (input_slot, expr) in self._expr_kwds.items():
            self._connect(module, expr, input_slot)

        self._module = module

    def validate(self):
        if self._valid is None:
            try:
                self._validate_args()
                self._validate_kwds()
                self._instanciate_module()
                self._valid = True
            except:
                self._valid = False
                raise
        if self._valid is False:
            raise ValidationError("Module not valid")
        return self._module

    def invalidate(self):
        self._valid = None

    def scheduler(self):
        return self._module.scheduler()

    def repipe(self, mod_name, out=None):
        mod_ = self.scheduler().module[mod_name]
        if isinstance(mod_, TableModule):
            from .table import TableExpr
            return TableExpr(
                type(mod_), (), dict(lazy=True), module=mod_, output_slot=out)
        return Expr(
            type(mod_), (), dict(lazy=True), module=mod_, output_slot=out)

    def fetch(self, mod_name, out=None):  # a simple alias for repipe
        return self.repipe(mod_name, out)

    def __or__(self, other):
        if other is None:
            return self
        if isinstance(other, Expr):
            return other
        ret = other._expr_class(other._module_class, (self, ) + other._args,
                                other._kwds)
        if other._repipe:
            return ret.repipe(other._repipe, out=other._repipe_out)
        return ret
