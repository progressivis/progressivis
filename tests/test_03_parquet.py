from __future__ import annotations

from . import ProgressiveTest
from progressivis.core import aio
from progressivis import ParquetLoader, Constant, PTable, get_dataset, Sink
from pyarrow import RecordBatch


class TestProgressiveLoadParquet(ProgressiveTest):
    def runit(self, module: ParquetLoader) -> int:
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
        assert module.result is not None
        self.assertEqual(len(module.result), 1000_000)

    def test_read_parquet_check_size(self) -> None:
        num_rows_list = []
        fixed_batch_size = 1234

        def _ff(bat: RecordBatch) -> RecordBatch:
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
        assert module.result is not None
        num_rows_set = set(num_rows_list[:-1])
        self.assertEqual(len(num_rows_set), 1)
        self.assertEqual(num_rows_set.pop(), fixed_batch_size)
        self.assertEqual(len(module.result), 1000_000)

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
        assert module.result is not None
        self.assertEqual(module.result.columns, columns)
        self.assertEqual(len(module.result), 1000_000)

    def test_read_multiple_parquet(self) -> None:
        s = self.scheduler()
        filenames = PTable(
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
        assert parquet.result is not None
        self.assertEqual(len(parquet.result), 60_000)


if __name__ == "__main__":
    ProgressiveTest.main()
