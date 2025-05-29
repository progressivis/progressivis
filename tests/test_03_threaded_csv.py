from __future__ import annotations

from . import ProgressiveTest, skipIf
from progressivis import Sink, ThreadedCSVLoader, get_dataset
from progressivis.core import aio
import os
import sys

TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"


@skipIf(os.getenv("CI") or sys.version_info < (3, 13),
        "for now, works only with python >= 3.13")
class TestProgressiveLoadCSV(ProgressiveTest):
    def runit(self, module: ThreadedCSVLoader) -> int:
        module.run(1)
        assert module.result is not None
        table = module.result
        self.assertFalse(table is None)
        _ = len(table)
        cnt = 2

        while not module.is_zombie():
            module.run(cnt)
            cnt += 1
            # s = module.trace_stats(max_runs=1)
            table = module.result
            _ = len(table)
            # print "Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), ln)
            # self.assertEqual(ln-l, len(df[df[module.UPDATE_COLUMN]==module.last_update()]))
            # l = ln
        _ = module.trace_stats(max_runs=1)
        # print("Done. Run time: %gs, loaded %d rows" % (s['duration'][-1], len(module.result)))
        return cnt

    def test_read_csv(self) -> None:
        s = self.scheduler()
        module = ThreadedCSVLoader(
            get_dataset("bigfile"), header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        assert module.result is not None
        self.assertEqual(len(module.result), 1000000)

    def test_read_csv_taxis(self) -> None:
        s = self.scheduler()
        module = ThreadedCSVLoader(
            TAXI_FILE, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        assert module.result is not None
        self.assertEqual(len(module.result), 12748987)


if __name__ == "__main__":
    ProgressiveTest.main()
