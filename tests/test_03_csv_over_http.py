from . import ProgressiveTest, skipIf

from multiprocessing import Process
import time
import os

from RangeHTTPServer import RangeRequestHandler  # type: ignore

import http.server as http_srv

from progressivis.core import aio

from progressivis import CSVLoader, Sink, Constant, PTable
from progressivis.datasets import get_dataset, get_dataset_bz2, DATA_DIR

from typing import Any, Optional


BZ2 = "csv.bz2"
SLEEP = 1
PORT: int = 9090
HOST: str = "localhost"


class ThrottledReqHandler(RangeRequestHandler):  # type: ignore
    threshold = 10**6
    sleep_times = 3

    def copyfile(self, src: Any, dest: Any) -> None:
        buffer_size = 1024 * 16
        sleep_times = ThrottledReqHandler.sleep_times
        if not self.range:
            cnt = 0
            while True:
                data = src.read(buffer_size)
                if not data:
                    break
                cnt += len(data)
                if sleep_times and cnt > ThrottledReqHandler.threshold:
                    time.sleep(1)
                    sleep_times -= 1
                dest.write(data)
        else:
            RangeRequestHandler.copyfile(self, src, dest)


def _close(module: CSVLoader) -> None:
    try:
        assert module.parser
        module.parser._input._stream.close()
    except Exception:
        pass


def run_throttled_server(port: int = PORT, threshold: int = 10**6) -> None:
    _ = get_dataset("smallfile")
    _ = get_dataset("bigfile")
    _ = get_dataset_bz2("smallfile")
    _ = get_dataset_bz2("bigfile")
    os.chdir(DATA_DIR)
    ThrottledReqHandler.threshold = threshold
    http_srv.test(HandlerClass=ThrottledReqHandler, port=port)  # type: ignore


def make_url(name: str, ext: str = "csv") -> str:
    return "http://{host}:{port}/{name}.{ext}".format(
        host=HOST, port=PORT, name=name, ext=ext
    )


def run_simple_server() -> None:
    _ = get_dataset("smallfile")
    _ = get_dataset("bigfile")
    _ = get_dataset_bz2("smallfile")
    _ = get_dataset_bz2("bigfile")
    os.chdir(DATA_DIR)
    import RangeHTTPServer.__main__  # type: ignore

    RangeHTTPServer.__main__


@skipIf(os.getenv("CI"), "cannot run an HTTP local server anymore on CI ...")
class TestProgressiveLoadCSVOverHTTP(ProgressiveTest):
    def setUp(self) -> None:
        super(TestProgressiveLoadCSVOverHTTP, self).setUp()
        self._http_proc: Optional[Process] = None

    def tearDown(self) -> None:
        if self._http_proc is not None:
            try:
                self._http_proc.terminate()
                time.sleep(SLEEP)
            except Exception:
                pass

    def test_01_read_http_csv_no_crash(self) -> None:
        p = Process(target=run_simple_server, args=())
        p.start()
        self._http_proc = p
        time.sleep(SLEEP)
        s = self.scheduler()
        module = CSVLoader(
            make_url("bigfile"), index_col=False, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        _close(module)
        assert module.result is not None
        self.assertEqual(len(module.result), 1000000)

    def test_02_read_http_csv_crash_recovery(self) -> None:
        p = Process(target=run_throttled_server, args=(PORT, 10**7))
        p.start()
        self._http_proc = p
        time.sleep(SLEEP)
        s = self.scheduler()
        module = CSVLoader(
            make_url("bigfile"), index_col=False, header=None, scheduler=s, timeout=0.01
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        _close(module)
        assert module.result is not None
        self.assertEqual(len(module.result), 1000000)

    def test_03_read_multiple_csv_crash_recovery(self) -> None:
        p = Process(target=run_throttled_server, args=(PORT, 10**6))
        p.start()
        self._http_proc = p
        time.sleep(SLEEP)
        s = self.scheduler()
        filenames = PTable(
            name="file_names",
            dshape="{filename: string}",
            data={"filename": [make_url("smallfile"), make_url("smallfile")]},
        )
        cst = Constant(table=filenames, scheduler=s)
        csv = CSVLoader(index_col=False, header=None, scheduler=s, timeout=0.01)
        csv.input.filenames = cst.output.result
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = csv.output.result
        aio.run(csv.start())
        _close(csv)
        assert csv.result is not None
        self.assertEqual(len(csv.result), 60000)

    def test_04_read_http_csv_bz2_no_crash(self) -> None:
        p = Process(target=run_simple_server, args=())
        p.start()
        self._http_proc = p
        time.sleep(SLEEP)
        s = self.scheduler()
        module = CSVLoader(
            make_url("bigfile", ext=BZ2), index_col=False, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        _close(module)
        assert module.result is not None
        self.assertEqual(len(module.result), 1000000)

    def test_05_read_http_csv_bz2_crash_recovery(self) -> None:
        p = Process(target=run_throttled_server, args=(PORT, 10**7))
        p.start()
        self._http_proc = p
        time.sleep(SLEEP)
        s = self.scheduler()
        module = CSVLoader(
            make_url("bigfile", ext=BZ2),
            index_col=False,
            header=None,
            scheduler=s,
            timeout=0.01,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        _close(module)
        assert module.result is not None
        self.assertEqual(len(module.result), 1000000)

    def test_06_read_multiple_csv_bz2_crash_recovery(self) -> None:
        p = Process(target=run_throttled_server, args=(PORT, 10**6))
        p.start()
        self._http_proc = p
        time.sleep(SLEEP)
        s = self.scheduler()
        filenames = PTable(
            name="file_names",
            dshape="{filename: string}",
            data={
                "filename": [
                    make_url("smallfile", ext=BZ2),
                    make_url("smallfile", ext=BZ2),
                ]
            },
        )
        cst = Constant(table=filenames, scheduler=s)
        csv = CSVLoader(index_col=False, header=None, scheduler=s, timeout=0.01)
        csv.input.filenames = cst.output.result
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = csv.output.result
        aio.run(csv.start())
        _close(csv)
        assert csv.result is not None
        self.assertEqual(len(csv.result), 60000)


if __name__ == "__main__":
    ProgressiveTest.main()
