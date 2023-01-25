from .core import Expr
from .table import PDataExpr, Pipeable
from progressivis.io.csv_loader import CSVLoader
from progressivis import Print


load_csv = Pipeable(PDataExpr, CSVLoader)
echo = Pipeable(Expr, Print)
