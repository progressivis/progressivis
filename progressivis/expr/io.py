from .core import Expr
from .table import PTableExpr, Pipeable
from progressivis.io.csv_loader import CSVLoader
from progressivis import Print


load_csv = Pipeable(PTableExpr, CSVLoader)
echo = Pipeable(Expr, Print)
