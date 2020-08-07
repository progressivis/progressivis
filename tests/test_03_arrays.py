from . import ProgressiveTest, skip, skipIf

from progressivis.core import aio
from progressivis import Print
from progressivis.arrays import Unary, Binary, unary_dict, binary_dict
import progressivis.arrays as arr
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
    def test_unary2(self):
        s = self.scheduler()
        random = RandomTable(10, rows=100000, scheduler=s)
        module = Unary(np.log, columns=['_3', '_5', '_7'], scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.log(random.table().to_array()[:, [2, 4, 6]])
        res2 = module.table().to_array()
        self.assertEqual(module.name, "unary_1")
        self.assertTrue(np.allclose(res1, res2))

    def _t_impl(self, cls, ufunc, mod_name):
        print("Testing", mod_name)
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

def add_un_test(k, ufunc):
    cls = k.capitalize()
    mod_name = k+'_1'
    def _f(self_):
        TestUnary._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)
    setattr(TestUnary, 'test_'+k, _f)
    
    
for k, ufunc in unary_dict.items():
    add_un_test(k, ufunc)

class TestBinary(ProgressiveTest):
    def test_binary(self):
        s = self.scheduler()
        random1 = RandomTable(3, rows=100000, scheduler=s)
        random2 = RandomTable(3, rows=100000, scheduler=s)
        from progressivis.table.dshape import dshape_projection

        module = Binary(np.add, scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.add(random1.table().to_array(),
                      random2.table().to_array())
        res2 = module.table().to_array()
        self.assertEqual(module.name, "binary_1")
        self.assertTrue(np.allclose(res1, res2))

    def test_binary2(self):
        s = self.scheduler()
        random1 = RandomTable(10, rows=100000, scheduler=s)
        random2 = RandomTable(10, rows=100000, scheduler=s)        
        module = Binary(np.add, columns=['_3', '_5', '_7'], scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.add(random1.table().to_array()[:, [2, 4, 6]],
                      random2.table().to_array()[:, [2, 4, 6]])
        res2 = module.table().to_array()
        self.assertEqual(module.name, "binary_1")
        self.assertTrue(np.allclose(res1, res2))

    def test_binary3(self):
        s = self.scheduler()
        random1 = RandomTable(10, rows=100000, scheduler=s)
        random2 = RandomTable(10, rows=100000, scheduler=s)        
        module = Binary(np.add, columns=['_3', '_5', '_7'],
                        columns2=['_4', '_6', '_8'], scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.add(random1.table().to_array()[:, [2, 4, 6]],
                      random2.table().to_array()[:, [3, 5, 7]])
        res2 = module.table().to_array()
        self.assertEqual(module.name, "binary_1")
        self.assertTrue(np.allclose(res1, res2))

    def _t_impl(self, cls, ufunc, mod_name):
        print("Testing", mod_name)
        s = self.scheduler()
        random1 = RandomTable(3, rows=1000000, scheduler=s)
        random2 = RandomTable(3, rows=1000000, scheduler=s)
        from progressivis.table.dshape import dshape_projection
        module = cls(scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = ufunc(random1.table().to_array(),
                      random2.table().to_array())
        res2 = module.table().to_array()
        self.assertEqual(module.name, mod_name)
        self.assertTrue(np.allclose(res1, res2))

def add_bin_test(k, ufunc):
    cls = k.capitalize()
    mod_name = k+'_1'
    def _f(self_):
        TestBinary._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)
    setattr(TestBinary, 'test_'+k, _f)
    
for k, ufunc in binary_dict.items():
     add_bin_test(k, ufunc)

