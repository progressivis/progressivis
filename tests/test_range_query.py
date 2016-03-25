import unittest

from progressivis import Module, Scheduler, Constant, Print, Select, Every
from progressivis.stats import RandomTable, Min, Max
from progressivis.core import RangeQuery
from progressivis.core.utils import last_row
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset


import pandas as pd
import numpy as np

def print_len(x):
    if x is not None:
        print len(x)

class TestRangeQuery(unittest.TestCase):
    def test_range_query(self):
        s=Scheduler()
        #table = CSVLoader(get_dataset('bigfile'), index_col=False,header=None, force_valid_ids=True,scheduler=s)
        table = RandomTable(['_1', '_2', '_3', '_4', '_5'], rows=10000, throttle=1000, scheduler=s)
        # min = Min(scheduler=s)
        # min.input.df = table.output.df
        # max = Max(scheduler=s)
        # max.input.df = table.output.df
        min_df = pd.DataFrame({'_1': [np.nan], '_2': [np.nan], '_3': [0.2], '_4': [0.1], '_5':[0.7]})
        min_value = Constant(min_df, scheduler=s)
        max_df = pd.DataFrame({'_1': [np.nan], '_2': [0.8], '_3': [0.9], '_4': [0.3], '_5':[np.nan]})
        max_value = Constant(max_df, scheduler=s)
        range_query = RangeQuery(scheduler=s)
        range_query.create_dependent_modules(table, 'df', min_value=min_value, max_value=max_value)
        # range_query.input.min = min.output.df
        # range_query.input.min_value = min_value.output.df
        # range_query.input.max = max.output.df
        # range_query.input.max_value = max_value.output.df
        pr=Print(id='print', scheduler=s)
        pr.input.df = range_query.output.query
        select=Select(scheduler=s)
        select.input.df = table.output.df
        select.input.query = range_query.output.query
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.df = select.output.df
        
        s.start()

        out_df  = last_row(range_query.get_data('df'), remove_update=True)
        out_min = last_row(range_query.get_data('min'), remove_update=True)
        out_max = last_row(range_query.get_data('max'), remove_update=True)
        print out_df
        print out_min
        print out_max
        self.assertTrue((out_min<  out_max).all())
        self.assertTrue((last_row(min_df)[['_3', '_4', '_5']]==out_min[['_3', '_4', '_5']]).all())
        self.assertTrue((last_row(max_df)[['_2', '_3', '_4']]==out_max[['_2', '_3', '_4']]).all())

        json = range_query.to_json()
        print json['ranges']
        self.assertTrue(json['ranges'] is not None)
        self.assertTrue(len(json['ranges']) == 5)
        self.assertTrue(all([range['out_min'] >= range['in_min'] and range['out_max'] <= range['in_max'] for range in json['ranges']]))


if __name__ == '__main__':
    unittest.main()
