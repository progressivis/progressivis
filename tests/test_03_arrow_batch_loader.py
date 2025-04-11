from __future__ import annotations
import duckdb
import pyarrow.parquet as pq
from . import ProgressiveTest
from progressivis.core.api import Sink
from progressivis.core import aio

from progressivis import ArrowBatchLoader, get_dataset


class TestArrowBatchLoader(ProgressiveTest):
    def test_read_csv(self) -> None:
        con = duckdb.connect(database=":memory:")
        file_name = get_dataset("bigfile_parquet")
        pq_file = pq.ParquetFile(file_name)
        n_rows = pq_file.metadata.num_rows
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


if __name__ == "__main__":
    ProgressiveTest.main()

"""
con = duckdb.connect(database=":memory:")
con.execute(f"SELECT * FROM read_parquet('{get_dataset("bigfile_parquet")}')")
reader = con.fetch_record_batch(1000)
while True:
    reader.read_next_batch()
#import pdb;pdb.set_trace()
#print(con.show())
"""
