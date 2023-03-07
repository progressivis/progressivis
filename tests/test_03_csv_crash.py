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
from progressivis.core import aio, Sink, Scheduler
from typing import cast, Optional, List

BZ2 = "csv.bz2"
GZ = "csv.gz"
XZ = "csv.xz"

PORT = 8000
HOST = "localhost"
SLEEP = 10

# IS_PERSISTENT = False


def _close(module: CSVLoader) -> None:
    try:
        assert module.parser is not None
        module.parser._input._stream.close()
    except Exception:
        pass


async def sleep_then_stop(s: Scheduler, t: float) -> None:
    await aio.sleep(t)
    await s.stop()
    # trace_after_stop(s)


def trace_after_stop(s: Scheduler) -> None:
    t = cast(CSVLoader, s["csv_loader_1"]).result
    assert t is not None
    print("crashed when len(_table) ==", len(t), "last_id:", t._last_id)
    i = t._last_id
    row = t.loc[i - 1, :]
    assert row is not None
    print("border row i:", row.to_dict())
    row = t.loc[i, :]
    assert row is not None
    print("border row i+1:", row.to_dict())


def make_url(name: str, ext: str = "csv") -> str:
    return "http://{host}:{port}/{name}.{ext}".format(
        host=HOST, port=PORT, name=name, ext=ext
    )


def run_simple_server() -> None:
    _ = get_dataset("smallfile")
    _ = get_dataset("bigfile")
    _ = get_dataset_bz2("smallfile")
    _ = get_dataset_bz2("bigfile")
    _ = get_dataset_gz("smallfile")
    _ = get_dataset_gz("bigfile")
    os.chdir(DATA_DIR)
    import RangeHTTPServer.__main__  # type: ignore

    assert RangeHTTPServer.__main__


BIGFILE_DF = pd.read_csv(filepath_or_buffer=get_dataset("bigfile"), header=None, usecols=[0])  # type: ignore


class _HttpSrv:
    def __init__(self) -> None:
        _HttpSrv.start(self)

    def stop(self) -> None:
        if self._http_proc is not None:
            try:
                self._http_proc.terminate()
                time.sleep(SLEEP)
            except Exception:
                pass

    def start(self) -> None:
        p = Process(target=run_simple_server, args=())
        p.start()
        self._http_proc = p
        time.sleep(SLEEP)

    def restart(self) -> None:
        self.stop()
        self.start()


# IS_PERSISTENT = False
class ProgressiveLoadCSVCrashRoot(ProgressiveTest):
    _http_srv: Optional[_HttpSrv] = None

    def setUp(self) -> None:
        super().setUp()
        cleanup_temp_dir()
        init_temp_dir_if()
        # if self._http_srv is None:
        #    self._http_srv =  _HttpSrv()

    def tearDown(self) -> None:
        super().tearDown()
        # TestProgressiveLoadCSVCrash.cleanup()
        if self._http_srv is not None:
            try:
                self._http_srv.stop()
            except Exception:
                pass
        cleanup_temp_dir()

    def get_tag(self) -> int:
        return id(self._http_srv)


# IS_PERSISTENT = False
class TestProgressiveLoadCSVCrash1(ProgressiveLoadCSVCrashRoot):
    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_01_read_http_csv_with_crash(self) -> None:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 1000_000)
        col = module.result.loc[:, 0]
        assert col is not None
        arr1 = col.to_array().reshape(-1)
        arr2 = BIGFILE_DF.loc[:, 0].values
        self.assertTrue(np.allclose(arr1, cast(List[float], arr2)))

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_01_read_http_csv_with_crash_and_counter(self) -> None:
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
        assert csv.result is not None
        assert counter.result is not None
        self.assertEqual(len(csv.result), 1000_000)
        self.assertEqual(counter.result["counter"].loc[0], 1000_000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_02_read_http_csv_bz2_with_crash(self) -> None:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 1000_000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_03_read_http_multi_csv_no_crash(self) -> None:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 60_000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_04_read_http_multi_csv_bz2_no_crash(self) -> None:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 60_000)


class TestProgressiveLoadCSVCrash2(ProgressiveLoadCSVCrashRoot):
    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_05_read_http_multi_csv_with_crash(self) -> None:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 2000_000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_06_read_http_multi_csv_bz2_with_crash(self) -> None:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 2000_000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_07_read_multi_csv_file_no_crash(self) -> None:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 60_000)


class TestProgressiveLoadCSVCrash3(ProgressiveLoadCSVCrashRoot):
    def _tst_08_read_multi_csv_file_compress_no_crash(self, files: List[str]) -> None:
        s = self.scheduler()
        module = CSVLoader(
            files, index_col=False, header=None, scheduler=s
        )  # , save_context=False)
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        assert module.result is not None
        self.assertEqual(len(module.result), 60_000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_08_read_multi_csv_file_bz2_no_crash(self) -> None:
        files = [get_dataset_bz2("smallfile")] * 2
        return self._tst_08_read_multi_csv_file_compress_no_crash(files)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_08_read_multi_csv_file_gz_no_crash(self) -> None:
        files = [get_dataset_gz("smallfile")] * 2
        return self._tst_08_read_multi_csv_file_compress_no_crash(files)

    @skip("Too slow ...")
    def test_08_read_multi_csv_file_lzma_no_crash(self) -> None:
        files = [get_dataset_lzma("smallfile")] * 2
        return self._tst_08_read_multi_csv_file_compress_no_crash(files)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_09_read_multi_csv_file_with_crash(self) -> None:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 2000_000)

    def _tst_10_read_multi_csv_file_compress_with_crash(
        self, file_list: List[str], tag: str
    ) -> None:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 2000_000)

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_10_read_multi_csv_file_bz2_with_crash(self) -> None:
        file_list = [get_dataset_bz2("bigfile")] * 2
        self._tst_10_read_multi_csv_file_compress_with_crash(file_list, "t10_1")

    @skipIf(not IS_PERSISTENT, "transient storage, test skipped")
    def test_10_read_multi_csv_file_gzip_with_crash(self) -> None:
        file_list = [get_dataset_gz("bigfile")] * 2
        self._tst_10_read_multi_csv_file_compress_with_crash(file_list, "t10_2")

    @skip("Too slow ...")
    def test_10_read_multi_csv_file_lzma_with_crash(self) -> None:
        file_list = [get_dataset_lzma("bigfile")] * 2
        self._tst_10_read_multi_csv_file_compress_with_crash(file_list, "t10_3")


if __name__ == "__main__":
    ProgressiveTest.main()
