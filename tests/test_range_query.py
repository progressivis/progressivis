import unittest

from progressivis import Module, Scheduler, Constant, Print
from progressivis.io import Variable
from progressivis.stats import RandomTable, Min, Max
from progressivis.core import RangeQuery
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset


import pandas as pd
import numpy as np

class TestRangeQuery(unittest.TestCase):
    def test_range_query(self):
        s=Scheduler()
        #table = CSVLoader(get_dataset('bigfile'), index_col=False,header=None, force_valid_ids=True,scheduler=s)
        table = RandomTable(['_1', '_2', '_3', '_4', '_5'], rows=1000000, scheduler=s)
        min = Min(scheduler=s)
        min.input.df = table.output.df
        max = Max(scheduler=s)
        max.input.df = table.output.df
        min_df = pd.DataFrame({'_1': [np.nan], '_2': [np.nan], '_3': [0.2], '_4': [0.1], '_5':[0.7]})
        min_value = Constant(min_df, scheduler=s)
        max_df = pd.DataFrame({'_1': [np.nan], '_2': [0.8], '_3': [0.9], '_4': [0.3], '_5':[np.nan]})
        max_value = Constant(max_df, scheduler=s)
        range_query = RangeQuery(scheduler=s)
        range_query.input.min = min.output.df
        range_query.input.min_value = min_value.output.df
        range_query.input.max = max.output.df
        range_query.input.max_value = max_value.output.df
        pr=Print(id='print', scheduler=s)
        pr.input.inp = range_query.output.query
        s.start()

        out_df  = Module.last_row(range_query.get_data('df'), remove_update=True)
        out_min = Module.last_row(range_query.get_data('min'), remove_update=True)
        out_max = Module.last_row(range_query.get_data('max'), remove_update=True)
        print out_df
        print out_min
        print out_max
        self.assertTrue((out_min<  out_max).all())
        self.assertTrue((Module.last_row(min_df)[['_3', '_4', '_5']]==out_min[['_3', '_4', '_5']]).all())
        self.assertTrue((Module.last_row(max_df)[['_2', '_3', '_4']]==out_max[['_2', '_3', '_4']]).all())


if __name__ == '__main__':
    unittest.main()
