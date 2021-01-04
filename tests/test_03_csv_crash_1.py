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
from ._csv_crash_utils import *
from ._csv_crash_utils import _HttpSrv

#IS_PERSISTENT = False   
class TestProgressiveLoadCSVCrash(ProgressiveLoadCSVCrashRoot):
    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_01_read_http_csv_with_crash(self):
        #if TRAVIS: return
        self._http_srv =  _HttpSrv()
        s=self.scheduler()
        url = make_url('bigfile')
        module=CSVLoader(url, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        sts = sleep_then_stop(s, 1)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s=self.scheduler(clean=True)
        module=CSVLoader(url, recovery=True, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 1000000)
        arr1 = module.table().loc[:, 0].to_array().reshape(-1)
        arr2 = BIGFILE_DF.loc[:, 0].values
        #import pdb;pdb.set_trace()        
        self.assertTrue(np.allclose(arr1, arr2))

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_01_read_http_csv_with_crash_and_counter(self):
        #if TRAVIS: return
        self._http_srv =  _HttpSrv()
        s=self.scheduler()
        url = make_url('bigfile')
        module=CSVLoader(url, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        sts = sleep_then_stop(s, 1)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s=self.scheduler(clean=True)
        csv=CSVLoader(url, recovery=True, index_col=False, header=None, scheduler=s)
        counter = Counter(scheduler=s)
        counter.input.table = csv.output.table
        self.assertTrue(csv.table() is None)
        aio.run(s.start())
        self.assertEqual(len(csv.table()), 1000000)
        self.assertEqual(counter.table()['counter'].loc[0], 1000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_02_read_http_csv_bz2_with_crash(self):
        #if TRAVIS: return
        self._http_srv =  _HttpSrv()
        s=self.scheduler()
        url = make_url('bigfile', ext=BZ2)
        module=CSVLoader(url, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        sts = sleep_then_stop(s, 4)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s=self.scheduler(clean=True)
        module=CSVLoader(url, recovery=True, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 1000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_03_read_http_multi_csv_no_crash(self):
        #if TRAVIS: return
        self._http_srv =  _HttpSrv()
        s=self.scheduler()
        module=CSVLoader([make_url('smallfile'),make_url('smallfile')], index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 60000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_04_read_http_multi_csv_bz2_no_crash(self):
        #if TRAVIS: return
        self._http_srv =  _HttpSrv()
        s=self.scheduler()
        module=CSVLoader([make_url('smallfile', ext=BZ2)]*2, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 60000)


if __name__ == '__main__':
    ProgressiveTest.main()
