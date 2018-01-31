from . import ProgressiveTest

from progressivis import Scheduler
from progressivis.table.table import Table

import numpy as np
import pandas as pd


class TestTableEval(ProgressiveTest):
    def setUp(self):
        super(TestTableEval, self).setUp()        
        self.scheduler = Scheduler.default
    def test_filtering(self):
        t = Table('table_filtering', dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100,size=20)
        t['a'] = ivalues
        fvalues = np.random.rand(20)*100
        t['b'] = fvalues
        df = pd.DataFrame(t.to_dict())
        
        def small_fun(expr, r):
            te = t.eval(expr, result_object=r)
            dfe = df.eval(expr)
            self.assertTrue(np.array_equal(te['a'], df[dfe]['a']))
            self.assertTrue(np.allclose(te['b'], df[dfe]['b']))
        def small_fun_ne(expr):
            r = 'raw_numexpr'
            te = t.eval(expr, result_object=r)
            dfe = df.eval(expr)
            self.assertTrue(np.array_equal(te, dfe.values))
        small_fun_ne('(a>10) & (a <80)')
        small_fun_ne('(b>10) & (b <80)')
        small_fun_ne('a>=b')
        small_fun('(a>10) & (a <80)', 'table')
        small_fun('(b>10) & (b <80)', 'table')
        small_fun('a>=b', 'table')
        small_fun('(a>10) & (a <80)', 'view')
    def test_assign(self):
        t = Table('table_eval_assign', dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100,size=20)
        t['a'] = ivalues
        fvalues = np.random.rand(20)*100
        t['b'] = fvalues
        df = pd.DataFrame(t.to_dict())
        t2 = t.eval('a = a+2*b', inplace=False)
        df2 = df.eval('a = a+2*b', inplace=False)
        self.assertTrue(np.allclose(t2['a'], df2['a']))
        self.assertTrue(np.allclose(t2['b'], df2['b']))
        t.eval('b = a+2*b', inplace=True)
        df.eval('b = a+2*b', inplace=True)
        self.assertTrue(np.allclose(t['a'], df['a']))
        self.assertTrue(np.allclose(t['b'], df['b']))        
    def test_user_dict(self):
        t = Table('table_eval_assign', dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100,size=20)
        t['a'] = ivalues
        fvalues = np.random.rand(20)*100
        t['b'] = fvalues
        df = pd.DataFrame(t.to_dict())
        t2 = t.eval('a = a+2*b', inplace=False)
        df2 = df.eval('x = a.loc[3]+2*b.loc[3]', inplace=False)
        print(df2.x)
        #self.assertTrue(np.allclose(t2['a'], df2['a']))
        #self.assertTrue(np.allclose(t2['b'], df2['b']))
        #t.eval('b = a+2*b', inplace=True)
        #df.eval('b = a+2*b', inplace=True)
        #self.assertTrue(np.allclose(t['a'], df['a']))
        #self.assertTrue(np.allclose(t['b'], df['b']))        
