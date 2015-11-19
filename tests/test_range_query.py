import unittest

from progressivis import *
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
        min_df = pd.DataFrame({'_1': [np.nan], '_2': [np.nan], '_3': [0.2], '_4': [0.00001], '_5':[0.7]})
        min_value = Variable(min_df, scheduler=s)
        max_df = pd.DataFrame({'_1': [np.nan], '_2': [0.8], '_3': [0.999], '_4': [0.3], '_5':[np.nan]})
        max_value = Variable(max_df, scheduler=s)
        range_query = RangeQuery(scheduler=s)
        range_query.input.min = min.output.df
        range_query.input.min_value = min_value.output.df
        range_query.input.max = max.output.df
        range_query.input.max_value = max_value.output.df
        pr=Print(id='print', scheduler=s)
        pr.input.inp = range_query.output.query
        s.start()

if __name__ == '__main__':
    unittest.main()
