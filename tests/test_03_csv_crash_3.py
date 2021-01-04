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
from ._csv_crash_utils import _HttpSrv, _close

#IS_PERSISTENT = False   
class TestProgressiveLoadCSVCrash(ProgressiveLoadCSVCrashRoot):
    def _tst_08_read_multi_csv_file_compress_no_crash(self, files):
        s=self.scheduler()
        module=CSVLoader(files, index_col=False, header=None, scheduler=s)#, save_context=False)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 60000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_08_read_multi_csv_file_bz2_no_crash(self):
        files = [get_dataset_bz2('smallfile')]*2
        return self._tst_08_read_multi_csv_file_compress_no_crash(files)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_08_read_multi_csv_file_gz_no_crash(self):
        files = [get_dataset_gz('smallfile')]*2
        return self._tst_08_read_multi_csv_file_compress_no_crash(files)

    @skip("Too slow ...")
    def test_08_read_multi_csv_file_lzma_no_crash(self):
        files = [get_dataset_lzma('smallfile')]*2
        return self._tst_08_read_multi_csv_file_compress_no_crash(files)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_09_read_multi_csv_file_with_crash(self):
        s=self.scheduler()
        file_list = [get_dataset('bigfile'), get_dataset('bigfile')]
        module=CSVLoader(file_list, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        sts = sleep_then_stop(s, 2)
        aio.run_gather(s.start(), sts)
        _close(module)
        s=self.scheduler(clean=True)
        module=CSVLoader(file_list, recovery=True, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 2000000)

    def _tst_10_read_multi_csv_file_compress_with_crash(self, file_list):
        s=self.scheduler()
        module=CSVLoader(file_list, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        sts = sleep_then_stop(s, 2)
        aio.run_gather(s.start(), sts)
        _close(module)
        s=self.scheduler(clean=True)
        module=CSVLoader(file_list, recovery=True, index_col=False, header=None, scheduler=s)
        self.assertTrue(module.table() is None)
        aio.run(s.start())
        self.assertEqual(len(module.table()), 2000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_10_read_multi_csv_file_bz2_with_crash(self):
        file_list = [get_dataset_bz2('bigfile')]*2
        self._tst_10_read_multi_csv_file_compress_with_crash(file_list)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_10_read_multi_csv_file_gzip_with_crash(self):
        file_list = [get_dataset_gz('bigfile')]*2
        self._tst_10_read_multi_csv_file_compress_with_crash(file_list)

    @skip("Too slow ...")
    def test_10_read_multi_csv_file_lzma_with_crash(self):
        file_list = [get_dataset_lzma('bigfile')]*2
        self._tst_10_read_multi_csv_file_compress_with_crash(file_list)

if __name__ == '__main__':
    ProgressiveTest.main()
