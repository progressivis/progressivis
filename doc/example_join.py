from progressivis import Sink, Scheduler, ParquetLoader, CSVLoader, Join

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
