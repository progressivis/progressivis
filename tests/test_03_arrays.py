from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.arrays import Unary, Log, Log2, Log10
from progressivis.stats import RandomTable
import numpy as np

class TestUnary(ProgressiveTest):
    def test_unary(self):
        s = self.scheduler()
        random = RandomTable(10, rows=100000, scheduler=s)
        module = Unary(np.log, scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.log(random.table().to_array())
        res2 = module.table().to_array()
        self.assertEqual(module.name, "unary_1")
        self.assertTrue(np.allclose(res1, res2))
    def _t_impl(self, cls, ufunc, mod_name):
        s = self.scheduler()
        random = RandomTable(10, rows=100000, scheduler=s)
        module = cls(scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = ufunc(random.table().to_array())
        res2 = module.table().to_array()
        self.assertEqual(module.name, mod_name)
        self.assertTrue(np.allclose(res1, res2))

    def t_est_log(self):
        s = self.scheduler()
        random = RandomTable(10, rows=100000, scheduler=s)
        module = Log(scheduler=s)
        module.input.table = random.output.table
        print(module.name)
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.log(random.table().to_array())
        res2 = module.table().to_array()
        self.assertEqual(module.name, "log_1")
        self.assertTrue(np.allclose(res1, res2))
    def test_log(self):
        self._t_impl(Log, np.log, 'log_1')
    def test_log2(self):
        self._t_impl(Log2, np.log2, 'log2_1')
    def test_log10(self):        
        self._t_impl(Log10, np.log10, 'log10_1')        
        
