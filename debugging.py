import pandas as pd
import numpy as np
import tempfile as tf
import os
from progressivis import Scheduler, Print
from progressivis.io import SimpleCSVLoader, DynVar
from progressivis.stats import Histogram2D, Min, Max
from progressivis.datasets import get_dataset
from progressivis.vis import StatsExtender
from progressivis.table import Table
from progressivis.table.constant import Constant
from progressivis.utils.psdict import PsDict
from progressivis.stats.scaling import MinMaxScaler
from progressivis_nb_widgets.nbwidgets import DataViewer
from progressivis.datasets import get_dataset
from progressivis.core import aio

s = Scheduler.default = Scheduler()

PREFIX = '../../nyc-taxi/'

SUFFIX = '.bz2'

URLS = [
    PREFIX+'yellow_tripdata_2015-01.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-02.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-03.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-04.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-05.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-06.csv'+SUFFIX,
]
num_cols = ['pickup_longitude', 'pickup_latitude','dropoff_longitude',
        'dropoff_latitude', 'trip_distance', 'RateCodeID']
cols = num_cols #+ ['store_and_fwd_flag']
FILENAMES = pd.DataFrame({'filename': URLS})
CST = Constant(Table('filenames', data=FILENAMES), scheduler=s)
CSV = SimpleCSVLoader(index_col=False, skipinitialspace=True,
                usecols=cols, scheduler=s, throttle=100)

CSV.input.filenames = CST.output[0]
#pr=Print(proc=lambda x: None, scheduler=s)
#pr.input[0] = CSV.output.result
stext = StatsExtender(usecols=cols)
some_cols = ['pickup_longitude','pickup_latitude','dropoff_longitude'] #,'dropoff_latitude']

stext.create_dependent_modules(CSV, hist=some_cols, min_=True,
                              max_=True,
                              var=True, distinct=True, 
                              corr=True)
aio.run(s.start())
