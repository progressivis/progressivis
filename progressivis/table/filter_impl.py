
import numexpr as ne

from .mod_impl import ModuleImpl
from progressivis.core.utils import indices_len, fix_loc
from progressivis.core.bitmap import bitmap

import numpy as np


_len = indices_len
_fix = fix_loc


class _Selection(object):
    def __init__(self, values=None):
        self._values = bitmap([]) if values is None else values

    def add(self, values):
        self._values.update(values)

    def remove(self, values):
        self._values = self._values - bitmap(values)


class FilterImpl(ModuleImpl):
    def __init__(self, expr, user_dict=None):
        super(FilterImpl, self).__init__()
        self._expr = expr
        self._user_dict = user_dict
        self._table = None
        self.result = _Selection()

    def resume(self, created=None, updated=None, deleted=None, fastpath=False):
        if updated:
            raise ValueError('Update in filtering is not implemented')
        if deleted:
            self.result.remove(deleted)
        if created:
            res = ne.evaluate(self._expr, local_dict=self._user_dict)
            if res.dtype != 'bool':
                raise ValueError('expr must be a conditional expression!')
            if fastpath:
                ix = np.where(res)[0] + created.start
            else:
                ix = np.where(res)[0]
            self.result.add(ix)

    def start(self, table, created=None, updated=None, deleted=None):
        self._table = table
        # TODO optimize the other cases
        fastpath = updated is None and deleted is None \
            and isinstance(created, slice)
        if fastpath:
            self._user_dict = {key: table[key].values[created]
                               for key in table.columns}
        else:
            self._user_dict = {key: table[key].values
                               for key in table.columns}
        self.is_started = True
        return self.resume(created, updated, deleted, fastpath)
