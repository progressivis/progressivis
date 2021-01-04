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
    def test_05_read_http_multi_csv_with_crash(self):
        #if TRAVIS: return
        self._http_srv =  _HttpSrv()
        s=self.scheduler()
        url_list = [make_url('bigfile'),make_url('bigfile')]
        module=CSVLoader(url_list, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        sts = sleep_then_stop(s, 2)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s=self.scheduler(clean=True)
        module=CSVLoader(url_list, recovery=True, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 2000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_06_read_http_multi_csv_bz2_with_crash(self):
        #if TRAVIS: return
        self._http_srv =  _HttpSrv()
        s=self.scheduler()
        url_list = [make_url('bigfile', ext=BZ2)]*2
        module=CSVLoader(url_list, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        sts = sleep_then_stop(s, 2)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s=self.scheduler(clean=True)
        module=CSVLoader(url_list, recovery=True, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 2000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_07_read_multi_csv_file_no_crash(self):
        s=self.scheduler()
        module=CSVLoader([get_dataset('smallfile'), get_dataset('smallfile')], index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 60000)

if __name__ == '__main__':
    ProgressiveTest.main()
