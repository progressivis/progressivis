from . import ProgressiveTest

import datashape as ds

from collections import OrderedDict
from progressivis import Scheduler
from progressivis.table.pytables import PyTableView
from progressivis.io.csv_loader import CSVLoader
from progressivis.datasets import get_dataset

import numpy as np
import pandas as pd
import tables as pt

class AB(pt.IsDescription):
    idnumber  = pt.Int64Col()
    a  = pt.Int64Col()
    b = pt.Float32Col()


def create_pt(name, desc):
    h5file = pt.open_file("/tmp/progressivis_test_pt.h5", mode = "w", title = "Pytables test file")
    group = h5file.create_group("/", 'gab', 'AB class group')
    pt_ = h5file.create_table(group, name, AB, desc)
    line = pt_.row
    ivalues = np.random.randint(100,size=20)
    fvalues = np.random.rand(20)
    for i in range(20):
        line['a'] = ivalues[i]
        line['b'] = fvalues[i]
        line.append()
    pt_.flush()
    return pt_, ivalues, fvalues

    
class TestPyTableView(ProgressiveTest):
    def setUp(self):
        super(TestPyTableView, self).setUp()
        self.scheduler = Scheduler.default

    def test_at(self):
        pt_, ivalues, fvalues = create_pt('ptab', 'Py table')
        view = PyTableView(pt_)
        self.assertEqual(type(view), PyTableView)
        self.assertEqual(view.at[2, 'a'], ivalues[2])
        self.assertEqual(view.iat[2, 0], ivalues[2])        


