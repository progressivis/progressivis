from . import ProgressiveTest, skip


import logging
import sys
import six
from progressivis.datasets import (get_dataset, get_dataset_bz2,
                                       get_dataset_gz, get_dataset_lzma,
                                       DATA_DIR)

class TestLoadDatasets(ProgressiveTest):
    def test_load_smallfile(self):
        _ = get_dataset('smallfile')
    def test_load_bigfile(self):        
        _ = get_dataset('bigfile')
    def test_load_smallfile_bz2(self):
        _ = get_dataset_bz2('smallfile')
    def test_load_bigfile_bz2(self):        
        _ = get_dataset_bz2('bigfile')
    def test_load_smallfile_gz(self):
        _ = get_dataset_gz('smallfile')
    def test_load_bigfile_gz(self):        
        _ = get_dataset_gz('bigfile')
    def test_load_smallfile_lzma(self):
        if six.PY2:
            return
        _ = get_dataset_lzma('smallfile')
    @skip("Too slow ...")
    def test_load_bigfile_lzma(self):
        if six.PY2:
            return        
        _ = get_dataset_lzma('bigfile')
        
