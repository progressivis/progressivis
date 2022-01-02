from multiprocessing import Process
import time
import os

import numpy as np
import pandas as pd

from . import ProgressiveTest, skip, skipIf
from progressivis.io import CSVLoader
from progressivis.datasets import (
    get_dataset,
    get_dataset_bz2,
    get_dataset_gz,
    get_dataset_lzma,
    DATA_DIR,
)
from progressivis.stats.counter import Counter
from progressivis.storage import IS_PERSISTENT
from progressivis.storage import cleanup_temp_dir, init_temp_dir_if
from progressivis.core import aio, Sink


BZ2 = "csv.bz2"
GZ = "csv.gz"
XZ = "csv.xz"

PORT = 8000
HOST = "localhost"
SLEEP = 10

# IS_PERSISTENT = False


def _close(module):
    try:
        module.parser._input._stream.close()
    except Exception:
        pass


async def sleep_then_stop(s, t):
    await aio.sleep(t)
    await s.stop()
    # trace_after_stop(s)


def trace_after_stop(s):
    t = s.modules()["csv_loader_1"]._table
    print("crashed when len(_table) ==", len(t), "last_id:", t._last_id)
    i = t._last_id
    print("border row i:", t.loc[i - 1, :].to_dict())
    print("border row i+1:", t.loc[i, :].to_dict())


def make_url(name, ext="csv"):
    return "http://{host}:{port}/{name}.{ext}".format(
        host=HOST, port=PORT, name=name, ext=ext
    )


def run_simple_server():
    _ = get_dataset("smallfile")
    _ = get_dataset("bigfile")
    _ = get_dataset_bz2("smallfile")
    _ = get_dataset_bz2("bigfile")
    _ = get_dataset_gz("smallfile")
    _ = get_dataset_gz("bigfile")
    # if six.PY3:
    #    _ = get_dataset_lzma('smallfile')
    #    _ = get_dataset_lzma('bigfile')
    os.chdir(DATA_DIR)
    import RangeHTTPServer.__main__  # type: ignore

    assert RangeHTTPServer.__main__


BIGFILE_DF = pd.read_csv(filepath_or_buffer=get_dataset("bigfile"), header=None, usecols=[0])  # type: ignore


class _HttpSrv(object):
    def __init__(self):
        _HttpSrv.start(self)

    def stop(self):
        if self._http_proc is not None:
            try:
                self._http_proc.terminate()
                time.sleep(SLEEP)
            except Exception:
                pass

    def start(self):
        p = Process(target=run_simple_server, args=())
        p.start()
        self._http_proc = p
        time.sleep(SLEEP)

    def restart(self):
        self.stop()
        self.start()


# IS_PERSISTENT = False
class ProgressiveLoadCSVCrashRoot(ProgressiveTest):
    _http_srv = None

    def setUp(self):
        super().setUp()
        # self._http_srv = None
        cleanup_temp_dir()
        init_temp_dir_if()
        # if self._http_srv is None:
        #    self._http_srv =  _HttpSrv()

    def tearDown(self):
        super().tearDown()
        # TestProgressiveLoadCSVCrash.cleanup()
        if self._http_srv is not None:
            try:
                self._http_srv.stop()
            except Exception:
                pass
        cleanup_temp_dir()

    def get_tag(self):
        return id(self._http_srv)


# IS_PERSISTENT = False
class TestProgressiveLoadCSVCrash1(ProgressiveLoadCSVCrashRoot):
    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_01_read_http_csv_with_crash(self):
        self._http_srv = _HttpSrv()
        tag = self.get_tag()
        s = self.scheduler()
        url = make_url("bigfile")
        module = CSVLoader(
            url, index_col=False, recovery_tag=tag, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        sts = sleep_then_stop(s, 2)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s = self.scheduler(clean=True)
        module = CSVLoader(
            url,
            recovery=True,
            recovery_tag=tag,
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 1000000)
        arr1 = module.result.loc[:, 0].to_array().reshape(-1)
        arr2 = BIGFILE_DF.loc[:, 0].values
        # import pdb;pdb.set_trace()
        self.assertTrue(np.allclose(arr1, arr2))

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_01_read_http_csv_with_crash_and_counter(self):
        self._http_srv = _HttpSrv()
        tag = self.get_tag()
        s = self.scheduler()
        url = make_url("bigfile")
        module = CSVLoader(
            url, index_col=False, recovery_tag=tag, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        sts = sleep_then_stop(s, 2)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s = self.scheduler(clean=True)
        csv = CSVLoader(
            url,
            recovery=True,
            index_col=False,
            recovery_tag=tag,
            header=None,
            scheduler=s,
        )
        counter = Counter(scheduler=s)
        counter.input[0] = csv.output.result
        self.assertTrue(csv.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = counter.output.result
        aio.run(s.start())
        self.assertEqual(len(csv.result), 1000000)
        self.assertEqual(counter.result["counter"].loc[0], 1000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_02_read_http_csv_bz2_with_crash(self):
        self._http_srv = _HttpSrv()
        tag = self.get_tag()
        s = self.scheduler()
        url = make_url("bigfile", ext=BZ2)
        module = CSVLoader(
            url, index_col=False, recovery_tag=tag, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        sts = sleep_then_stop(s, 5)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s = self.scheduler(clean=True)
        module = CSVLoader(
            url,
            recovery=True,
            recovery_tag=tag,
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 1000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_03_read_http_multi_csv_no_crash(self):
        self._http_srv = _HttpSrv()
        s = self.scheduler()
        module = CSVLoader(
            [make_url("smallfile"), make_url("smallfile")],
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 60000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_04_read_http_multi_csv_bz2_no_crash(self):
        self._http_srv = _HttpSrv()
        s = self.scheduler()
        module = CSVLoader(
            [make_url("smallfile", ext=BZ2)] * 2,
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 60000)


class TestProgressiveLoadCSVCrash2(ProgressiveLoadCSVCrashRoot):
    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_05_read_http_multi_csv_with_crash(self):
        self._http_srv = _HttpSrv()
        tag = self.get_tag()
        s = self.scheduler()
        url_list = [make_url("bigfile"), make_url("bigfile")]
        module = CSVLoader(
            url_list, index_col=False, recovery_tag=tag, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        sts = sleep_then_stop(s, 3)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s = self.scheduler(clean=True)
        module = CSVLoader(
            url_list,
            recovery=True,
            recovery_tag=tag,
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 2000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_06_read_http_multi_csv_bz2_with_crash(self):
        self._http_srv = _HttpSrv()
        tag = self.get_tag()
        s = self.scheduler()
        url_list = [make_url("bigfile", ext=BZ2)] * 2
        module = CSVLoader(
            url_list, index_col=False, recovery_tag=tag, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        sts = sleep_then_stop(s, 3)
        aio.run_gather(s.start(), sts)
        self._http_srv.restart()
        s = self.scheduler(clean=True)
        module = CSVLoader(
            url_list,
            recovery=True,
            recovery_tag=tag,
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 2000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_07_read_multi_csv_file_no_crash(self):
        s = self.scheduler()
        module = CSVLoader(
            [get_dataset("smallfile"), get_dataset("smallfile")],
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 60000)


class TestProgressiveLoadCSVCrash3(ProgressiveLoadCSVCrashRoot):
    def _tst_08_read_multi_csv_file_compress_no_crash(self, files):
        s = self.scheduler()
        module = CSVLoader(
            files, index_col=False, header=None, scheduler=s
        )  # , save_context=False)
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 60000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_08_read_multi_csv_file_bz2_no_crash(self):
        files = [get_dataset_bz2("smallfile")] * 2
        return self._tst_08_read_multi_csv_file_compress_no_crash(files)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_08_read_multi_csv_file_gz_no_crash(self):
        files = [get_dataset_gz("smallfile")] * 2
        return self._tst_08_read_multi_csv_file_compress_no_crash(files)

    @skip("Too slow ...")
    def test_08_read_multi_csv_file_lzma_no_crash(self):
        files = [get_dataset_lzma("smallfile")] * 2
        return self._tst_08_read_multi_csv_file_compress_no_crash(files)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_09_read_multi_csv_file_with_crash(self):
        s = self.scheduler()
        tag = "t9"
        file_list = [get_dataset("bigfile"), get_dataset("bigfile")]
        module = CSVLoader(
            file_list, index_col=False, recovery_tag=tag, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        sts = sleep_then_stop(s, 3)
        aio.run_gather(s.start(), sts)
        _close(module)
        s = self.scheduler(clean=True)
        module = CSVLoader(
            file_list,
            recovery=True,
            recovery_tag=tag,
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 2000000)

    def _tst_10_read_multi_csv_file_compress_with_crash(self, file_list, tag):
        s = self.scheduler()
        module = CSVLoader(
            file_list, index_col=False, recovery_tag=tag, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        sts = sleep_then_stop(s, 4)
        aio.run_gather(s.start(), sts)
        _close(module)
        s = self.scheduler(clean=True)
        module = CSVLoader(
            file_list,
            recovery=True,
            recovery_tag=tag,
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 2000000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_10_read_multi_csv_file_bz2_with_crash(self):
        file_list = [get_dataset_bz2("bigfile")] * 2
        self._tst_10_read_multi_csv_file_compress_with_crash(file_list, "t10_1")

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_10_read_multi_csv_file_gzip_with_crash(self):
        file_list = [get_dataset_gz("bigfile")] * 2
        self._tst_10_read_multi_csv_file_compress_with_crash(file_list, "t10_2")

    @skip("Too slow ...")
    def test_10_read_multi_csv_file_lzma_with_crash(self):
        file_list = [get_dataset_lzma("bigfile")] * 2
        self._tst_10_read_multi_csv_file_compress_with_crash(file_list, "t10_3")


if __name__ == "__main__":
    ProgressiveTest.main()
