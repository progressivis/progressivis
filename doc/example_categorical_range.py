from progressivis import Sink, Scheduler, CSVLoader, ConstDict, PDict
from progressivis.table.categorical_query import CategoricalQuery

scheduler = Scheduler()
with scheduler:
    csv = CSVLoader("path/to/your/data.csv", scheduler=scheduler)
    # NB: data.csv contains a categorical column named "category"
    query = CategoricalQuery(column="category", scheduler=scheduler)
    query.create_dependent_modules(input_module=csv)
    ct = ConstDict(PDict({"only": ["A", "C"]}), scheduler=scheduler)
    query.input.choice = ct.output.result
    sink = Sink(scheduler=scheduler)
    sink.input.inp = query.output.result
