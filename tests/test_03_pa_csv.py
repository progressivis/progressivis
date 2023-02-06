from __future__ import annotations
from pyarrow.csv import ReadOptions
from . import ProgressiveTest
from progressivis.core import aio, Sink
from progressivis.io import PACSVLoader
from progressivis.table.constant import Constant
from progressivis.table.table import PTable
from progressivis.datasets import get_dataset  # , RandomBytesIO
from progressivis.core.utils import RandomBytesIO

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.module import Module


class TestProgressiveLoadCSV(ProgressiveTest):
    def runit(self, module: Module) -> int:
        module.run(1)
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
        module = PACSVLoader(
            get_dataset("bigfile"),
            read_options=ReadOptions(autogenerate_column_names=True),
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        # import pdb;pdb.set_trace()
        self.assertEqual(len(module.result), 1000000)

    def test_read_fake_csv(self) -> None:
        s = self.scheduler()
        module = PACSVLoader(
            RandomBytesIO(cols=30, rows=1000000),
            read_options=ReadOptions(autogenerate_column_names=True),
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 1000000)

    def test_read_multiple_csv(self) -> None:
        s = self.scheduler()
        filenames = PTable(
            name="file_names",
            dshape="{filename: string}",
            data={"filename": [get_dataset("smallfile"), get_dataset("smallfile")]},
        )
        cst = Constant(table=filenames, scheduler=s)
        csv = PACSVLoader(
            read_options=ReadOptions(autogenerate_column_names=True), scheduler=s
        )
        csv.input.filenames = cst.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = csv.output.result
        aio.run(csv.start())
        self.assertEqual(len(csv.result), 60000)

    def test_read_multiple_fake_csv(self) -> None:
        s = self.scheduler()
        filenames = PTable(
            name="file_names2",
            dshape="{filename: string}",
            data={
                "filename": [
                    "buffer://fake1?cols=10&rows=30000",
                    "buffer://fake2?cols=10&rows=30000",
                ]
            },
        )
        cst = Constant(table=filenames, scheduler=s)
        csv = PACSVLoader(
            read_options=ReadOptions(autogenerate_column_names=True), scheduler=s
        )
        csv.input.filenames = cst.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = csv.output.result
        aio.run(csv.start())
        self.assertEqual(len(csv.result), 60000)


if __name__ == "__main__":
    ProgressiveTest.main()
