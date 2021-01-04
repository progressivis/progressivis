from . import ProgressiveTest, skip, skipIf
from progressivis.io import CSVLoader
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.datasets import (get_dataset, get_dataset_bz2,
                                       get_dataset_gz,
                                       get_dataset_lzma, DATA_DIR)
from progressivis.core.utils import RandomBytesIO
from progressivis.stats.counter import Counter
from progressivis.storage import IS_PERSISTENT
from progressivis.storage import  cleanup_temp_dir, init_temp_dir_if
from progressivis.core import aio
#import logging, sys
from multiprocessing import Process
import time, os
import requests
from requests.packages.urllib3.exceptions import ReadTimeoutError
from requests.exceptions import ConnectionError


from RangeHTTPServer import RangeRequestHandler
import shutil
import numpy as np
import pandas as pd

import http.server as http_srv

BZ2 = 'csv.bz2'
GZ = 'csv.gz'
XZ = 'csv.xz'
#TRAVIS = os.getenv("TRAVIS")
PORT = 8000
HOST = 'localhost'
SLEEP = 10

#IS_PERSISTENT = False

def _close(module):
    try:
        module.parser._input._stream.close()
    except:
        pass

async def sleep_then_stop(s, t):
    await aio.sleep(t)
    await s.stop()
    t = s.modules()['csv_loader_1']._table
    print("crashed when len(_table) ==", len(t), "last_id:", t._last_id)
    i = t._last_id
    print("border row i:", t.loc[i-1,:].to_dict())
    print("border row i+1:", t.loc[i,:].to_dict())    
    #import pdb;pdb.set_trace()
    #print(s._run_list)

def make_url(name, ext='csv'):
    return 'http://{host}:{port}/{name}.{ext}'.format(host=HOST,
                                                        port=PORT,
                                                        name=name, ext=ext)

def run_simple_server():
    _ = get_dataset('smallfile')
    _ = get_dataset('bigfile')
    _ = get_dataset_bz2('smallfile')
    _ = get_dataset_bz2('bigfile')
    _ = get_dataset_gz('smallfile')
    _ = get_dataset_gz('bigfile')
    #if six.PY3:
    #    _ = get_dataset_lzma('smallfile')
    #    _ = get_dataset_lzma('bigfile')
    os.chdir(DATA_DIR)
    import RangeHTTPServer.__main__

BIGFILE_DF = pd.read_csv(get_dataset('bigfile'), header=None, usecols=[0])

class _HttpSrv(object):
    def __init__(self):
        _HttpSrv.start(self)

    def stop(self):
        if self._http_proc is not None:
            try:
                self._http_proc.terminate()
                time.sleep(SLEEP)
            except:
                pass

    def start(self):
        p = Process(target=run_simple_server, args=())
        p.start()
        self._http_proc = p
        time.sleep(SLEEP)

    def restart(self):
        self.stop()
        self.start()

#IS_PERSISTENT = False   
class ProgressiveLoadCSVCrashRoot(ProgressiveTest):
    _http_srv = None
    def setUp(self):
        super().setUp()        
        #self._http_srv = None
        cleanup_temp_dir()
        init_temp_dir_if()
        #if self._http_srv is None:
        #    self._http_srv =  _HttpSrv()

    def tearDown(self):
        super().tearDown()
        #TestProgressiveLoadCSVCrash.cleanup()
        if self._http_srv is not None:
            try:
                self._http_srv.stop()
            except:
                pass
        cleanup_temp_dir()
