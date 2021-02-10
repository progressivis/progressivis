from . import ProgressiveTest

import datashape as ds

from collections import OrderedDict
from progressivis import Scheduler
from progressivis.table.table import Table, BaseTable
#from progressivis.table.table_sliced import TableSlicedView
#from progressivis.table.table_selected import TableSelectedView
from progressivis.io.csv_loader import CSVLoader
from progressivis.datasets import get_dataset
from progressivis.storage import Group
from progressivis.core import aio
import numpy as np
import pandas as pd


class TestTable(ProgressiveTest):
    #pylint: disable=protected-access
    def setUp(self):
        super(TestTable, self).setUp()
        self.scheduler = Scheduler.default
        self.storagegroup = Group.default()


    def test_steps(self):
        self.create_table()
        self.fill_table()
        self.update_table()
        self.delete_table()
        self.examine_table()
        self.append_dataframe()
        self.append_direct()
        self.load_csv()
        self.fill_table()

    def test_loc_tableview(self):
        t = Table('table_loc', dshape="{a: int, b: float32}", create=True)
        t.resize(10)
        ivalues = np.random.randint(100,size=20)
        t['a'] = ivalues[:10]
        fvalues = np.random.rand(20)
        t['b'] = fvalues[:10]
        t.append({'a': ivalues[10:], 'b': fvalues[10:]})
        view = t.loc[2:11]
        self.assertEqual(type(view), BaseTable)
        self.assertTrue(np.array_equal(view._column(0)[:], ivalues[2:12]))
        view_view = view.loc[3:7]
        self.assertTrue(np.array_equal(view_view._column(0)[:], view._column(0)[3:7]))
        view_view = view.loc[3:6]
        self.assertTrue(np.array_equal(view_view._column(0)[:], view._column(0)[view.id_to_index(slice(3,6))]))
        table_view = view.loc[[3,4,6,9]]
        self.assertEqual(type(table_view),BaseTable)
        self.assertTrue(np.array_equal(table_view._column(0).values, view._column(0)[[3,4,6,9]]))
        table_view = view.loc[[3,4,6,9]]
        self.assertEqual(type(table_view),BaseTable)
        self.assertTrue(np.array_equal(table_view._column(0).values, view._column(0)[view.id_to_index([3,4,6,9])]))

    def test_set_loc(self):
        t = Table('table_set_loc', dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100,size=20)
        t['a'] = ivalues
        fvalues = np.random.rand(20)
        t['b'] = fvalues
        t.loc[3:6] = [1001, 1002]
        self.assertTrue(np.array_equal(t._column(0)[3:7], np.repeat(1001, 4)))
        self.assertTrue(np.array_equal(t._column(1)[3:7], np.repeat(1002, 4)))
        t.loc[3:7] = 1003
        self.assertTrue(np.array_equal(t._column(0)[3:8], np.repeat(1003, 5)))
        self.assertTrue(np.array_equal(t._column(1)[3:8], np.repeat(1003, 5)))
        t.loc[3:7,['a','b']] = [1004, 1005]
        self.assertTrue(np.array_equal(t._column(0)[3:8], np.repeat(1004, 5)))
        self.assertTrue(np.array_equal(t._column(1)[3:8], np.repeat(1005, 5)))
        t.loc[3:7,['a','b']] = [1006, 1007] # previous iloc test
        self.assertTrue(np.array_equal(t._column(0)[3:7], np.repeat(1006, 4)))
        self.assertTrue(np.array_equal(t._column(1)[3:7], np.repeat(1007, 4)))
        view = t.loc[2:11]
        view.loc[3:6] = [1008, 1009]
        self.assertTrue(np.array_equal(view._column(0)[
            view.id_to_index(slice(3,6))], np.repeat(1008, 4)))
        self.assertTrue(np.array_equal(view._column(1)[
            view.id_to_index(slice(3,6))], np.repeat(1009, 4)))
        self.assertTrue(np.array_equal(t._column(0)[3:7],
                                       np.repeat(1008, 4)))
        self.assertTrue(np.array_equal(t._column(1)[3:7],
                                       np.repeat(1009, 4)))
        view_view = view.loc[3:6]
        view_view.loc[3:6] = [1010, 1011]
        self.assertTrue(np.array_equal(view_view._column(0)[
            view_view.id_to_index(slice(3,6))], np.repeat(1010, 4)))
        self.assertTrue(np.array_equal(view_view._column(1)[
            view_view.id_to_index(slice(3,6))], np.repeat(1011, 4)))
        self.assertTrue(np.array_equal(t._column(0)[3:7], np.repeat(1010, 4)))
        self.assertTrue(np.array_equal(t._column(1)[3:7], np.repeat(1011, 4)))

    def test_at(self):
        t = Table('table_at', dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100,size=20)
        t['a'] = ivalues
        fvalues = np.random.rand(20)
        t['b'] = fvalues
        at_ = t.at[3,'a']
        self.assertEqual(at_, t._column(0)[3])
        iat_ = t.at[3, 1]
        self.assertEqual(iat_, t._column(1)[3])
        view = t.loc[2:11]
        at_ = view.at[3,'a']
        self.assertEqual(at_, view._column(0)[view.id_to_index(3)])
        iat_ = view.at[3, 1]
        self.assertEqual(iat_, view._column(1)[3])

    def test_set_at(self):
        t = Table('table_set_at', dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100,size=20)
        t['a'] = ivalues
        fvalues = np.random.rand(20)
        t['b'] = fvalues
        t.at[3, 'a'] = 1001
        self.assertEqual(t._column(0)[3], 1001)
        t.at[3, 'a'] = 1001
        self.assertEqual(t._column(0)[3], 1001)
        t.at[3, 0] = 1002
        self.assertEqual(t._column(0)[3], 1002)
        view = t.loc[2:11]
        view.loc[3, 'a'] = 1003
        self.assertEqual(view._column(0)[view.id_to_index(3)], 1003)
        self.assertEqual(t._column(0)[3], 1003)
        view_view = view.loc[3:6]
        view_view.at[3, 'a'] = 1004
        self.assertEqual(view_view._column(0)[view_view.id_to_index(3)], 1004)
        self.assertEqual(t._column(0)[3], 1004)
        view_view.at[2, 0] = 1005
        self.assertEqual(view_view._column(0)[2], 1005)
        self.assertEqual(t._column(0)[t.id_to_index(view_view.index_to_id(2))], 1005)

    def test_last(self):
        t = Table('table_last', dshape="{a: int, b: float32}", create=True)
        t.resize(10)
        ivalues = np.random.randint(100,size=10)
        t['a'] = ivalues
        fvalues = np.random.rand(10)
        t['b'] = fvalues
        last_ = list(t.last().values())
        self.assertEqual(last_, [t._column(0)[-1],t._column(1)[-1]])
        last_a = t.last('a')
        self.assertEqual(last_a, t._column(0)[-1])
        last_a_b = t.last(['a','b'])
        self.assertEqual(list(last_a_b),last_)

    def create_table(self):
        t = Table('table',
                  storagegroup=self.storagegroup,
                  dshape="{a: int, b: float32, c: string, d: 10*int}", create=True)
        self.assertTrue(t is not None)
        self.assertEqual(t.ncol, 4)
        col1 = t['a']
        col2 = t[0]
        self.assertTrue(col1 is col2)

        t = Table('table',
                  storagegroup=self.storagegroup,
                  dshape="{a: int, b: float32, c: string, d: 10*int}")
        self.assertTrue(t is not None)

        t = Table('table', storagegroup=self.storagegroup)
        self.assertEqual(t.dshape, ds.dshape("{a: int, b: float32, c: string, d: 10 * int}"))

        t2 = Table('bar_table',
                   dshape="{a: int64, b: float64}",
                   fillvalues={'a': -1}, create=True)
        self.assertEqual(t2.dshape, ds.dshape("{a: int64, b: float64}"))
        self.assertEqual(t2[0].fillvalue, -1)

    def fill_table(self):
        t = Table('table', storagegroup=self.storagegroup)
        self._fill_table(t)

    def _fill_table(self, t):
        # Try with a 10 elements Table
        t.resize(10)
        # Fill one column with a simple list
        ivalues = range(10)
        t['a'] = ivalues # Table._setitem_key
        icol = t['a'].value
        for i in range(len(ivalues)):
            self.assertEqual(ivalues[i], icol[i])

        ivalues = np.random.randint(100,size=10)
        t['a'] = ivalues
        icol = t['a'].value

        for i in range(len(ivalues)):
            self.assertEqual(ivalues[i], icol[i])
        t['b'] = ivalues
        fcol = t['b'].value

        for i in range(len(ivalues)):
            self.assertEqual(ivalues[i], fcol[i])

        fvalues = np.random.rand(10)
        t['b'] = fvalues
        fcol = t['b'].value
        for i in range(len(fvalues)):
            self.assertAlmostEqual(fvalues[i], fcol[i])

        #self.assertRaises(ValueError, t['a'] = values[1:])
        try:
            t['a'] = ivalues[1:]
        except ValueError:
            pass
        else:
            self.fail('ExpectedException not raised')
        # Fill multiple colums with
        ivalues = np.random.randint(100,size=10)
        fvalues = np.random.rand(10)
        t[['a', 'b']] = [ivalues, fvalues]
        icol = t['a'].value
        fcol = t['b'].value
        for i in range(len(fvalues)):
            self.assertEqual(ivalues[i], icol[i])
            self.assertAlmostEqual(fvalues[i], fcol[i])
        values = np.random.randint(100,size=(10, 2))
        t[['a', 'b']] = values
        icol = t['a'].value
        fcol = t['b'].value
        for i in range(len(fvalues)):
            self.assertEqual(values[i, 0], icol[i])
            self.assertEqual(values[i, 1], fcol[i])

        #self.assertRaises(ValueError, t[['a','b']] = values[1:])
        try:
            t[['a','b']] = values[:,1:]
        except TypeError: # h5py raises a TypeError
            pass
        except ValueError: # numpy would raise a ValueError
            pass
        #pylint: disable=broad-except
        except Exception as e:
            self.fail('Unexpected exception raised: %s'% e)
        else:
            self.fail('ExpectedException not raised')
        #f.close()

    def update_table(self):
        t = Table('table', storagegroup=self.storagegroup)
        self._update_table(t)

    def delete_table(self):
        t = Table('table', storagegroup=self.storagegroup)
        self._delete_table(t)

    def examine_table(self):
        t = Table('table', storagegroup=self.storagegroup)
        pass

    def _update_table(self, t):
        #pylint: disable=protected-access
        self.assertEqual(len(t),10)
        #t.scheduler._run_number = 1
        t['a'] = np.arange(10)
        #t.scheduler._run_number = 2
        t.loc[2:3, 'a'] = np.arange(2) # loc is inclusive
        v1 = t.loc[2:3, 'a']
        v11 = v1.loc[2,'a']
        v12 = v1.loc[2,:]
        v2 = t.loc[:, 'a']
        v3 = t.loc[:]
    def _delete_table(self, t):
        self.assertEqual(t.index_to_id(2), 2)
        a = t['a']
        self.assertEqual(a[2], a.fillvalue)
        del t.loc[2]
        with self.assertRaises(KeyError):
            c = t.loc[2]
            print(c)
        self.assertEqual(len(t), a.size-1)
        cnt = 0
        for row in t.iterrows():
            self.assertTrue('a' in row)
            cnt += 1
        self.assertEqual(len(t), cnt)

    def _delete_table2(self, t):
        with self.assertRaises(KeyError):
            c = t.loc[2]
            print(c)

        
    def append_dataframe(self):
        #pylint: disable=protected-access
        #self.scheduler._run_number = 1
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3], 'c': ['a', 'b', 'cd']})
        t = Table('table_2', data=df)
        self.assertEqual(len(t),len(df))
        for colname in df:
            coldf = df[colname]
            colt = t[colname]
            self.assertEqual(len(coldf), len(colt))
            self.assertTrue(np.all(coldf.values==colt.values))
        #self.scheduler._run_number = 2
        t.append(df)
        self.assertEqual(len(t),2*len(df))
        for colname in df:
            coldf = df[colname]
            colt = t[colname]
            self.assertEqual(2*len(coldf), len(colt))
            self.assertTrue(np.all(coldf==colt[len(df):len(t)]))

        #self.scheduler._run_number = 3
        t.append(t) # fun test
        self.assertEqual(len(t),4*len(df))
        for colname in df:
            coldf = df[colname]
            colt = t[colname]
            self.assertEqual(4*len(coldf), len(colt))
            self.assertTrue(np.all(colt[0:2*len(df)]==colt[2*len(df):len(t)]))



    def append_direct(self):
        #pylint: disable=protected-access
        d = OrderedDict([('a', [1, 2, 3]), ('b', [0.1, 0.2, 0.3]), ('c', ['a', 'b', 'cd'])])
        #print(dshape_extract(d))
        df = pd.DataFrame(d)
        #self.scheduler._run_number = 1
        t = Table('table_3', data=d)
        self.assertEqual(len(t),len(df))
        for colname in df:
            coldf = df[colname]
            colt = t[colname]
            self.assertEqual(len(coldf), len(colt))
            self.assertTrue(np.all(coldf==colt.values))

        #self.scheduler._run_number = 2
        t.append(d)
        self.assertEqual(len(t),2*len(df))
        for colname in df:
            coldf = df[colname]
            colt = t[colname]
            self.assertEqual(2*len(coldf), len(colt))
            self.assertTrue(np.all(coldf==colt[len(df):len(t)]))

        #self.scheduler._run_number = 3
        t.append(t) # fun test
        self.assertEqual(len(t),4*len(df))
        for colname in df:
            coldf = df[colname]
            colt = t[colname]
            self.assertEqual(4*len(coldf), len(colt))
            self.assertTrue(np.all(colt[0:2*len(df)]==colt[2*len(df):len(t)]))

    def load_csv(self):
        module=CSVLoader(filepath_or_buffer=get_dataset('smallfile'),
                         force_valid_ids=True,
                         index_col=False,
                         header=None,
                         scheduler=self.scheduler)
        self.assertTrue(module.result is None)
        aio.run(self.scheduler.start(persist=True))
        t = module.result
        self.assertFalse(t is None)
        self.assertEqual(len(t), 30000)
        df = pd.read_csv(filepath_or_buffer=get_dataset('smallfile'),
                         index_col=False,
                         header=None)
        for col in range(t.ncol):
            coldf = df[col]
            colt = t[col]
            self.assertTrue(np.all(coldf==colt.values))
        #print(t)

    def test_read_direct(self):
        t = Table('table_read_direct', dshape="{a: int, b: float32}", create=True)
        t.resize(10)
        ivalues = np.random.randint(100,size=10)
        t['a'] = ivalues
        fvalues = np.random.rand(10)
        t['b'] = fvalues
        a = t['a']
        jvalues = np.empty(10, dtype=a.dtype)
        a.read_direct(jvalues, np.s_[0:10], np.s_[0:10])
        self.assertTrue(np.all(ivalues==jvalues))
        b = t['b']
        gvalues = np.empty(10, dtype=b.dtype)
        b.read_direct(gvalues, np.s_[0:10], np.s_[0:10])
        self.assertTrue(np.allclose(fvalues, gvalues))

        a.read_direct(jvalues, np.s_[2:7], np.s_[5:10])
        self.assertTrue(np.all(ivalues[2:7]==jvalues[5:10]))
        b.read_direct(gvalues, np.s_[2:7], np.s_[5:10])
        self.assertTrue(np.allclose(fvalues[2:7], gvalues[5:10]))

    def test_to_array(self):
        t = Table('table_to_array', dshape="{a: int, b: float32, c: real}", create=True)
        t.resize(10)
        ivalues = np.random.randint(100,size=10)
        t['a'] = ivalues
        fvalues = np.random.rand(10)
        t['b'] = fvalues
        dvalues = np.random.rand(10)
        t['c'] = dvalues
        a = t['a']
        b = t['b']
        c = t['c']
        arr = t.to_array()
        self.assertEqual(arr.dtype, np.float64)
        self.assertEqual(arr.shape[0], t.nrow)
        self.assertEqual(arr.shape[1], t.ncol)
        self.assertTrue(np.allclose(a[:], arr[:, 0]))
        self.assertTrue(np.allclose(b[:], arr[:, 1]))
        self.assertTrue(np.allclose(c[:], arr[:, 2]))

        # Columns
        arr = t.to_array(columns=['a', 'b'])
        self.assertEqual(arr.dtype, np.float64)
        self.assertEqual(arr.shape[0], t.nrow)
        self.assertEqual(arr.shape[1], 2)
        self.assertTrue(np.allclose(a[:], arr[:, 0]))
        self.assertTrue(np.allclose(b[:], arr[:, 1]))

        # Keys
        key = slice(2,7)
        arr = t.to_array(key)
        key = t.id_to_index(key).to_slice_maybe() # slices contain their bounds
        self.assertEqual(arr.dtype, np.float64)
        self.assertEqual(arr.shape[0], key.stop-key.start)
        self.assertEqual(arr.shape[1], 3)
        self.assertTrue(np.allclose(a[key], arr[:, 0]))
        self.assertTrue(np.allclose(b[key], arr[:, 1]))
        self.assertTrue(np.allclose(c[key], arr[:, 2]))

        # Keys with fancy indexing
        key = [2,4,6,8]
        arr = t.to_array(key)
        indices = t.id_to_index(key) # slices contain their bounds
        self.assertEqual(arr.dtype, np.float64)
        self.assertEqual(arr.shape[0], len(indices))
        self.assertEqual(arr.shape[1], 3)
        self.assertTrue(np.allclose(a[indices], arr[:, 0]))
        self.assertTrue(np.allclose(b[indices], arr[:, 1]))
        self.assertTrue(np.allclose(c[indices], arr[:, 2]))

        #TODO more tests multidimensional columns and deleted rows

    def test_convert(self):
        arr = np.random.rand(10,5)

        t = Table.from_array(arr)
        self.assertIsNotNone(t)
        self.assertEqual(len(t.columns), arr.shape[1])
        self.assertEqual(t.columns, ['_1', '_2', '_3', '_4', '_5'])

        arr2 = t.to_array()
        self.assertTrue(np.allclose(arr, arr2))

        columns=['a', 'b', 'c']
        t = Table.from_array(arr, columns=columns, offsets=[0, 1, 3, 5])
        self.assertIsNotNone(t)
        self.assertEqual(len(t.columns), 3)
        self.assertEqual(t.columns, columns)


if __name__ == '__main__':
    ProgressiveTest.main()
