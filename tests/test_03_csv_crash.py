from __future__ import absolute_import
from . import ProgressiveTest
from progressivis.io import CSVLoader
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.datasets import get_dataset,get_dataset_bz2,  DATA_DIR
from progressivis.core.utils import RandomBytesIO
#import logging, sys
from multiprocessing import Process
import time, os
import requests
from requests.packages.urllib3.exceptions import ReadTimeoutError
from requests.exceptions import ConnectionError
from progressivis.core.utils import decorate, ModulePatch

from RangeHTTPServer import RangeRequestHandler

import six
import shutil

if six.PY3:
    import http.server as http_srv
else:
    import SimpleHTTPServer as http_srv

BZ2 = 'csv.bz2'


PORT = 8000
HOST = 'localhost'


class Patch1(ModulePatch):
    max_steps = 10000
    def before_run_step(self, m, *args, **kwargs):
        if m._table is not None and len(m._table) >Patch1.max_steps :
            print("Simulate a crash ...")
            m.scheduler().stop()
            #if hasattr(m._table, '__append'):
            #    return
            #m._table.__append = m._table.append

def make_url(name, ext='csv'):
    return 'http://{host}:{port}/{name}.{ext}'.format(host=HOST,
                                                        port=PORT,
                                                        name=name, ext=ext)

def run_simple_server():
    _ = get_dataset('smallfile')
    _ = get_dataset('bigfile')
    _ = get_dataset_bz2('smallfile')
    _ = get_dataset_bz2('bigfile')
    os.chdir(DATA_DIR)
    if six.PY2:
        import SimpleHTTPServer
        import RangeHTTPServer
        from RangeHTTPServer import RangeRequestHandler
        import sys
        sys.argv[1] = 8000
        SimpleHTTPServer.test(HandlerClass=RangeRequestHandler)
    else:
        import RangeHTTPServer.__main__
        
class TestProgressiveLoadCSVCrash(ProgressiveTest):
    def setUp(self):
        super(TestProgressiveLoadCSVCrash, self).setUp()        
        self._http_proc = None

    def tearDown(self):
        TestProgressiveLoadCSVCrash.cleanup()
        if self._http_proc is not None:
            try:
                self._http_proc.terminate()
                time.sleep(1)
            except:
                pass

    def test_01_read_http_csv_no_crash(self):
        p = Process(target=run_simple_server, args=())
        p.start()
        self._http_proc = p
        time.sleep(1)
        s=self.scheduler()
        module=CSVLoader(make_url('bigfile'), index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        decorate(s, Patch1("csv_loader_1"))
        s.start()
        s.join()
        module.parser._input._stream.close() # close the previous HTTP request
        #                              # necessary only because the
        #                              # SimpleHTTPServer is not multi-threaded
        s=self.scheduler()
        module=CSVLoader(make_url('bigfile'), recovery=True, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        s.start()
        s.join()
        t = module.table()

    def test_02_read_http_multi_csv_no_crash(self):
        p = Process(target=run_simple_server, args=())
        p.start()
        self._http_proc = p
        time.sleep(1)
        s=self.scheduler()
        module=CSVLoader([make_url('smallfile'),make_url('smallfile')], index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        #decorate(s, Patch1("csv_loader_1"))
        s.start()
        s.join()
        self.assertEqual(len(module.table()), 60000)

    def test_03_read_http_multi_csv_no_crash(self):
        p = Process(target=run_simple_server, args=())
        p.start()
        self._http_proc = p
        time.sleep(1)
        s=self.scheduler()
        url_list = [make_url('bigfile'),make_url('bigfile')]
        module=CSVLoader(url_list, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        Patch1.max_steps = 1200000
        decorate(s, Patch1("csv_loader_1"))
        s.start()
        s.join()
        module.parser._input._stream.close() # close the previous HTTP request
        #                              # necessary onle because the
        #                              # SimpleHTTPServer is not multi-threaded
        s=self.scheduler()
        module=CSVLoader(url_list, recovery=True, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        s.start()
        s.join()
        t = module.table()
        #import pdb;pdb.set_trace()
        #print("to csv ...")
        #t.to_csv('/tmp/bigfile2.csv')
        #print("... done")
        self.assertEqual(len(module.table()), 2000000)




if __name__ == '__main__':
    ProgressiveTest.main()
