from progressivis.core import Sink, Scheduler
from progressivis.io import ParquetLoader, CSVLoader
from progressivis.table.join import Join

PARQUET_FILE = "path/to/yellow_tripdata_2015-01.parquet"
CSV_URL = "path/to/taxi+_zone_lookup.csv"

scheduler = Scheduler()
with scheduler:
    parquet = ParquetLoader(PARQUET_FILE, scheduler=scheduler)
    csv = CSVLoader(CSV_URL, scheduler=scheduler)
    join = Join(how="inner", scheduler=scheduler)
    join.create_dependent_modules(
        related_module=parquet,
        primary_module=csv,
        related_on=["DOLocationID"],
        primary_on=["LocationID"]
    )
    sink = Sink(scheduler=scheduler)
    sink.input.inp = join.output.result
