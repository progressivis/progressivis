from . import ProgressiveTest
from progressivis.table.table import Table
#from progressivis.table.table_selected import TableSelectedView
from progressivis.core.bitmap import bitmap

import numpy as np
import pandas as pd

class TestToDict(ProgressiveTest):
    def test_to_dict(self):
        # index=[1,2,3,8,11],
        df = pd.DataFrame(data={'a': [1, 2, 3, 4, 5, 6, 7, 8],
                               'b': [10, 20, 30, 40, 50, 60, 70, 80],
                               'c': ['a', 'b', 'cd', 'ef', 'fg', 'gh', 'hi', 'ij']})
        t = Table(name=None, data=df)
        df = df.drop(df.index[[3,4]])
        del t.loc[[3,4]]
        #del t.loc[3]
        #print(df.to_dict(orient='index'))
        #print(df.to_dict(orient='records'))
        #print(t.to_dict(orient='index'))
        # orient : {'dict', 'list', 'split', 'rows', 'record', 'index'}
        self.assertEqual(df.to_dict(orient='dict'), t.to_dict(orient='dict'))
        self.assertEqual(df.to_dict(orient='list'), t.to_dict(orient='list'))
        self.assertEqual(df.to_dict(orient='split'), t.to_dict(orient='split'))
        self.assertEqual(df.to_dict(orient='records'), t.to_dict(orient='records'))
        self.assertEqual(df.to_dict(orient='index'), t.to_dict(orient='index'))

    def test_to_dict2(self):
        # index=[1,2,3,8,11],
        df = pd.DataFrame(data={'a': [1, 2, 3, 4, 5, 6, 7, 8],
                               'b': [10, 20, 30, 40, 50, 60, 70, 80],
                               'c': ['a', 'b', 'cd', 'ef', 'fg', 'gh', 'hi', 'ij']})
        t_ = Table(name=None, data=df)
        df = df.drop(df.index[[3,4]])
        sel = bitmap(t_.index) -bitmap([3,4])
        #del t.loc[[3,4]]
        t = t_.loc[sel, :] # TableSelectedView(t_, sel)
        #del t.loc[3]
        #print(df.to_dict(orient='records'))
        #print(df.to_dict(orient='records'))
        #print(t.to_dict(orient='records'))
        # orient : {'dict', 'list', 'split', 'rows', 'record', 'index'}
        self.assertEqual(df.to_dict(orient='dict'), t.to_dict(orient='dict'))
        self.assertEqual(df.to_dict(orient='list'), t.to_dict(orient='list'))
        self.assertEqual(df.to_dict(orient='split'), t.to_dict(orient='split'))
        self.assertEqual(df.to_dict(orient='records'), t.to_dict(orient='records'))
        self.assertEqual(df.to_dict(orient='index'), t.to_dict(orient='index'))
