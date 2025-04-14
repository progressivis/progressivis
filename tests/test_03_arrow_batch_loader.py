from __future__ import annotations
import duckdb
import pyarrow.parquet as pq
from . import ProgressiveTest
from progressivis.core.api import Sink
from progressivis.core import aio

from progressivis import ArrowBatchLoader, get_dataset


class TestArrowBatchLoader(ProgressiveTest):
    def test_read_parquet(self) -> None:
        con = duckdb.connect(database=":memory:")
        file_name = get_dataset("bigfile_parquet")
        n_rows = pq.ParquetFile(file_name).metadata.num_rows
        con.execute(f"SELECT * FROM read_parquet('{file_name}')")
        reader = con.fetch_record_batch(1000)
        s = self.scheduler()
        module = ArrowBatchLoader(
            reader=reader,
            n_rows=n_rows,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        assert module.result is not None
        self.assertEqual(len(module.result), n_rows)

    def test_read_csv(self) -> None:
        con = duckdb.connect(database=":memory:")
        file_name = get_dataset("bigfile")
        n_rows = 1_000_000
        con.execute(f"SELECT * FROM read_csv('{file_name}')")
        reader = con.fetch_record_batch(1000)
        s = self.scheduler()
        module = ArrowBatchLoader(
            reader=reader,
            n_rows=n_rows,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        assert module.result is not None
        self.assertEqual(len(module.result.columns), 30)
        self.assertEqual(len(module.result), n_rows)

    def test_read_csv_2_cols(self) -> None:
        con = duckdb.connect(database=":memory:")
        file_name = get_dataset("bigfile")
        n_rows = 1_000_000
        con.execute(f"SELECT column01, column02 FROM read_csv('{file_name}')")
        reader = con.fetch_record_batch(1000)
        s = self.scheduler()
        module = ArrowBatchLoader(
            reader=reader,
            n_rows=n_rows,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        assert module.result is not None
        self.assertEqual(len(module.result.columns), 2)
        self.assertEqual(len(module.result), n_rows)


if __name__ == "__main__":
    ProgressiveTest.main()
