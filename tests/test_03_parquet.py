from __future__ import annotations

from . import ProgressiveTest
from progressivis.core import aio, Sink
from progressivis.io import ParquetLoader
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.datasets import get_dataset


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.table.module import TableModule


class TestProgressiveLoadParquet(ProgressiveTest):
    def runit(self, module: TableModule) -> int:
        module.run(1)
        table = module.table
        self.assertFalse(table is None)
        _ = len(table)
        cnt = 2

        while not module.is_zombie():
            module.run(cnt)
            cnt += 1
            # s = module.trace_stats(max_runs=1)
            table = module.table
            _ = len(table)
            # print "Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), ln)
            # self.assertEqual(ln-l, len(df[df[module.UPDATE_COLUMN]==module.last_update()]))
            # l = ln
        _ = module.trace_stats(max_runs=1)
        # print("Done. Run time: %gs, loaded %d rows" % (s['duration'][-1], len(module.result)))
        return cnt

    def test_read_parquet(self) -> None:
        s = self.scheduler()
        module = ParquetLoader(
            get_dataset("bigfile_parquet"),
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.table), 1000000)

    def test_read_parquet_check_size(self) -> None:
        num_rows_list = []
        fixed_batch_size = 1234

        def _ff(bat):
            num_rows_list.append(bat.num_rows)
            return bat

        s = self.scheduler()
        module = ParquetLoader(
            get_dataset("bigfile_parquet"),
            batch_size=fixed_batch_size,
            filter_=_ff,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        num_rows_set = set(num_rows_list[:-1])
        self.assertEqual(len(num_rows_set), 1)
        self.assertEqual(num_rows_set.pop(), fixed_batch_size)
        self.assertEqual(len(module.table), 1000000)

    def test_read_parquet_with_cols(self) -> None:
        s = self.scheduler()
        columns = ["_1", "_5", "_15"]
        module = ParquetLoader(
            get_dataset("bigfile_parquet"),
            columns=columns,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(module.table.columns, columns)
        self.assertEqual(len(module.table), 1000000)

    def test_read_multiple_parquet(self) -> None:
        s = self.scheduler()
        filenames = Table(
            name="file_names",
            dshape="{filename: string}",
            data={
                "filename": [
                    get_dataset("smallfile_parquet"),
                    get_dataset("smallfile_parquet"),
                ]
            },
        )
        cst = Constant(table=filenames, scheduler=s)
        parquet = ParquetLoader(scheduler=s)
        parquet.input.filenames = cst.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = parquet.output.result
        aio.run(parquet.start())
        self.assertEqual(len(parquet.table), 60000)


if __name__ == "__main__":
    ProgressiveTest.main()
