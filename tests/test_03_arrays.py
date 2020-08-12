from . import ProgressiveTest, skip, skipIf

from progressivis.core import aio
from progressivis import Print
from progressivis.arrays import (Unary, Binary, Reduce,
                                 func2class_name,
                                 unary_module, make_unary,
                                 binary_module, make_binary,
                                 reduce_module, make_reduce,
                                 binary_dict_int_tst,
                                 unary_dict_gen_tst,
                                 binary_dict_gen_tst)
import progressivis.arrays as arr
#from progressivis.table.constant import Constant
from progressivis.stats import RandomTable, RandomDict
#from progressivis.utils.psdict import PsDict
import numpy as np

#@skip
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

def add_un_tst(k, ufunc):
    cls = func2class_name(k)
    mod_name = k+'_1'
    def _f(self_):
        TestUnary._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)
    setattr(TestUnary, 'test_'+k, _f)

for k, ufunc in unary_dict_gen_tst.items():
    add_un_tst(k, ufunc)

#@skip
class TestBinary(ProgressiveTest):
    def test_binary(self):
        s = self.scheduler()
        random1 = RandomTable(3, rows=100000, scheduler=s)
        random2 = RandomTable(3, rows=100000, scheduler=s)
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
        cols = 10
        random1 = RandomTable(cols, rows=100000, scheduler=s)
        random2 = RandomTable(cols, rows=100000, scheduler=s)
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

def add_bin_tst(c, k, ufunc):
    cls = func2class_name(k)
    mod_name = k+'_1'
    def _f(self_):
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)
    setattr(c, 'test_'+k, _f)
    
for k, ufunc in binary_dict_gen_tst.items():
     add_bin_tst(TestBinary, k, ufunc)

#@skip
class TestBinaryTD(ProgressiveTest):
    def test_binary(self):
        s = self.scheduler()
        cols = 3
        random1 = RandomTable(cols, rows=100000, scheduler=s)
        random2 =  RandomDict(cols, scheduler=s)
        module = Binary(np.add, scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.add(random1.table().to_array(),
                      np.array(list(random2.table().values())))
        res2 = module.table().to_array()
        self.assertEqual(module.name, "binary_1")
        self.assertTrue(np.allclose(res1, res2))

    def test_binary2(self):
        s = self.scheduler()
        cols = 10
        random1 = RandomTable(cols, rows=100000, scheduler=s)
        random2 = RandomDict(cols, scheduler=s)
        module = Binary(np.add, columns=['_3', '_5', '_7'], scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.add(random1.table().to_array()[:, [2, 4, 6]],
                      np.array(list(random2.table().values()))[[2, 4, 6]])
        res2 = module.table().to_array()
        self.assertEqual(module.name, "binary_1")
        self.assertTrue(np.allclose(res1, res2))

    def test_binary3(self):
        s = self.scheduler()
        cols = 10
        random1 = RandomTable(cols, rows=100000, scheduler=s)
        random2 = RandomDict(cols, scheduler=s)
        module = Binary(np.add, columns=['_3', '_5', '_7'],
                        columns2=['_4', '_6', '_8'], scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.add(random1.table().to_array()[:, [2, 4, 6]],
                      np.array(list(random2.table().values()))[[3, 5, 7]])
        res2 = module.table().to_array()
        self.assertEqual(module.name, "binary_1")
        self.assertTrue(np.allclose(res1, res2))

    def _t_impl(self, cls, ufunc, mod_name):
        print("Testing", mod_name)
        s = self.scheduler()
        cols = 3
        random1 = RandomTable(3, rows=1000000, scheduler=s)
        random2 = RandomDict(cols, scheduler=s)
        module = cls(scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = ufunc(random1.table().to_array(),
                      np.array(list(random2.table().values())))
        res2 = module.table().to_array()
        self.assertEqual(module.name, mod_name)
        self.assertTrue(np.allclose(res1, res2))

for k, ufunc in binary_dict_gen_tst.items():
     add_bin_tst(TestBinaryTD, k, ufunc)
#@skip
class TestReduce(ProgressiveTest):
    def test_reduce(self):
        s = self.scheduler()
        random = RandomTable(10, rows=100000, scheduler=s)
        module = Reduce(np.add, scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.add.reduce(random.table().to_array())
        res2 = np.array(list(module.table().values()))
        self.assertEqual(module.name, "reduce_1")
        self.assertTrue(np.allclose(res1, res2))
    def test_reduce2(self):
        s = self.scheduler()
        random = RandomTable(10,  rows=100000, scheduler=s)
        module = Reduce(np.add, columns=['_3', '_5', '_7'], scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.add.reduce(random.table().to_array()[:, [2, 4, 6]])
        res2 = np.array(list(module.table().values()))
        self.assertEqual(module.name, "reduce_1")
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
        res1 = getattr(ufunc, 'reduce')(random.table().to_array())
        res2 = np.array(list(module.table().values()))
        self.assertEqual(module.name, mod_name)
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


def add_reduce_tst(c, k, ufunc):
    cls = f"{func2class_name(k)}Reduce"
    mod_name = f'{k}_reduce_1'
    def _f(self_):
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)
    setattr(c, f'test_{k}', _f)
    
for k, ufunc in binary_dict_gen_tst.items():
    add_reduce_tst(TestReduce, k, ufunc)

class TestCustomFunctions(ProgressiveTest):

    def test_custom_unary(self):
        def dummy_unary(x):
            return (x+np.sin(x))/(x+np.cos(x))
        DummyUnary = make_unary(dummy_unary)
        s = self.scheduler()
        random = RandomTable(10, rows=100000, scheduler=s)
        module = DummyUnary(scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.array(module._ufunc(random.table().to_array()), dtype='float64')
        res2 = module.table().to_array()
        self.assertEqual(module.name, "dummy_unary_1")
        self.assertTrue(np.allclose(res1, res2))
    
    def test_custom_binary(self):
        def dummy_binary(x, y):
            return (x+np.sin(y))/(x+np.cos(y))
        DummyBinary = make_binary(dummy_binary)
        s = self.scheduler()
        random1 = RandomTable(3, rows=100000, scheduler=s)
        random2 = RandomTable(3, rows=100000, scheduler=s)
        module = DummyBinary(scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.array(module._ufunc(random1.table().to_array(),
                      random2.table().to_array()), dtype='float64')
        res2 = module.table().to_array()
        self.assertEqual(module.name, "dummy_binary_1")
        self.assertTrue(np.allclose(res1, res2))

    def test_custom_reduce(self):
        def dummy_binary(x, y):
            return (x+np.sin(y))/(x+np.cos(y))
        DummyBinaryReduce = make_reduce(dummy_binary)
        s = self.scheduler()
        random = RandomTable(10, rows=100000, scheduler=s)
        module = DummyBinaryReduce(scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.array(module._ufunc(random.table().to_array()), dtype='float64')
        res2 = np.array(list(module.table().values()))
        self.assertEqual(module.name, "dummy_binary_reduce_1")
        self.assertTrue(np.allclose(res1, res2))

from progressivis.arrays import Arccosh, Invert, BitwiseNot
class TestOtherUnaries(ProgressiveTest):
    def test_arccosh(self):
        #from progressivis.arrays import Arccosh
        module_name = "arccosh_1"
        print("Testing", module_name)
        s = self.scheduler()
        random = RandomTable(10, random=lambda x: np.random.rand(x)*10000.0,
                             rows=100000, scheduler=s)
        module = Arccosh(scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.arccosh(random.table().to_array())
        res2 = module.table().to_array()
        #print(res1)
        #print("=============================")
        #print(res2)
        self.assertEqual(module.name, module_name)
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))
    def test_invert(self):
        #from progressivis.arrays import Invert
        module_name = "invert_1"
        print("Testing", module_name)
        s = self.scheduler()
        random = RandomTable(10, random=lambda x: np.random.randint(100000, size=x),
                             dtype='int64',
                             rows=100000, scheduler=s)
        module = Invert(scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.invert(random.table().to_array())
        res2 = module.table().to_array()
        #print(res1)
        #print("=============================")
        #print(res2)
        self.assertEqual(module.name, module_name)
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))
    def test_bitwise_not(self):
        #from progressivis.arrays import Invert
        module_name = 'bitwise_not_1'
        print("Testing", module_name)
        s = self.scheduler()
        random = RandomTable(10, random=lambda x: np.random.randint(100000, size=x),
                             dtype='int64',
                             rows=100000, scheduler=s)
        module = BitwiseNot(scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.bitwise_not(random.table().to_array())
        res2 = module.table().to_array()
        #print(res1)
        #print("=============================")
        #print(res2)
        self.assertEqual(module.name, module_name)
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


from progressivis.arrays import Ldexp
class TestOtherBinaries(ProgressiveTest):
    def _t_impl(self, cls, ufunc, mod_name):
        print("Testing", mod_name)
        s = self.scheduler()
        random1 = RandomTable(3, rows=100000, scheduler=s, random=lambda x: np.random.randint(10, size=x),
                             dtype='int64')
        random2 = RandomTable(3, rows=100000, scheduler=s, random=lambda x: np.random.randint(10, size=x),
                             dtype='int64')
        module = cls(scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = ufunc(random1.table().to_array(),
                      random2.table().to_array())
        res2 = module.table().to_array()
        #print(res1)
        #print("=============================")
        #print(res2)
        self.assertEqual(module.name, mod_name)
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))
    def test_ldexp(self):
        cls, ufunc, mod_name = Ldexp, np.ldexp, 'ldexp_1'
        print("Testing", mod_name)
        s = self.scheduler()
        random1 = RandomTable(3, rows=100000, scheduler=s)
        random2 = RandomTable(3, rows=100000, scheduler=s, random=lambda x: np.random.randint(10, size=x),
                             dtype='int64')
        module = cls(scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = ufunc(random1.table().to_array(),
                      random2.table().to_array())
        res2 = module.table().to_array()
        #print(res1)
        #print("=============================")
        #print(res2)
        self.assertEqual(module.name, mod_name)
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

def add_other_bin_tst(c, k, ufunc):
    cls = func2class_name(k)
    mod_name = k+'_1'
    def _f(self_):
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)
    setattr(c, 'test_'+k, _f)
    
for k, ufunc in binary_dict_int_tst.items():
    if k == 'ldexp':
        continue
    add_other_bin_tst(TestOtherBinaries, k, ufunc)


class TestOtherReduces(ProgressiveTest):
    def _t_impl(self, cls, ufunc, mod_name):
        print("Testing", mod_name)
        s = self.scheduler()
        random = RandomTable(3, rows=100000, scheduler=s, random=lambda x: np.random.randint(10, size=x),
                             dtype='int64')
        module = cls(scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = getattr(ufunc, 'reduce')(random.table().to_array())
        res2 = np.array(list(module.table().values()))
        #print(res1)
        #print("=============================")
        #print(res2)
        self.assertEqual(module.name, mod_name)
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))
        self.assertEqual(module.name, mod_name)
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

def add_other_reduce_tst(c, k, ufunc):
    cls = f"{func2class_name(k)}Reduce"
    mod_name = f'{k}_reduce_1'
    def _f(self_):
        c._t_impl(self_, arr.__dict__[cls], ufunc, mod_name)
    setattr(c, f'test_{k}', _f)
    
for k, ufunc in binary_dict_int_tst.items():
    if k == 'ldexp':
        continue
    add_other_reduce_tst(TestOtherReduces, k, ufunc)

#for c in ["TestUnary", "TestBinary", "TestBinaryTD",
#          "TestReduce", "TestCustomFunctions",
#          "TestOtherUnaries", "TestOtherBinaries", "TestOtherReduces"]: del globals()[c]

class TestDecorators(ProgressiveTest):

    def test_decorator_unary(self):
        @unary_module
        def DummyUnary(x):
            return (x+np.sin(x))/(x+np.cos(x))
        s = self.scheduler()
        random = RandomTable(10, rows=100000, scheduler=s)
        module = DummyUnary(scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.array(module._ufunc(random.table().to_array()), dtype='float64')
        res2 = module.table().to_array()
        self.assertEqual(module.name, "dummy_unary_1")
        self.assertTrue(np.allclose(res1, res2))

    def test_decorator_binary(self):
        @binary_module
        def DummyBinary(x, y):
            return (x+np.sin(y))/(x+np.cos(y))
        s = self.scheduler()
        random1 = RandomTable(3, rows=100000, scheduler=s)
        random2 = RandomTable(3, rows=100000, scheduler=s)
        module = DummyBinary(scheduler=s)
        module.input.first = random1.output.table
        module.input.second = random2.output.table        
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.array(module._ufunc(random1.table().to_array(),
                      random2.table().to_array()), dtype='float64')
        res2 = module.table().to_array()
        self.assertEqual(module.name, "dummy_binary_1")
        self.assertTrue(np.allclose(res1, res2))

    def test_decorator_reduce(self):
        @reduce_module
        def DummyBinaryReduce(x, y):
            return (x+np.sin(y))/(x+np.cos(y))
        s = self.scheduler()
        random = RandomTable(10, rows=100000, scheduler=s)
        module = DummyBinaryReduce(scheduler=s)
        module.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = module.output.table
        aio.run(s.start())
        res1 = np.array(module._ufunc(random.table().to_array()), dtype='float64')
        res2 = np.array(list(module.table().values()))
        self.assertEqual(module.name, "dummy_binary_reduce_1")
        self.assertTrue(np.allclose(res1, res2))
