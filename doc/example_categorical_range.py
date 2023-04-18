from progressivis.core import Sink, Scheduler
from progressivis.io import CSVLoader
from progressivis.table.categorical_query import CategoricalQuery
from progressivis.table.constant import ConstDict
from progressivis.utils.psdict import PDict

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
