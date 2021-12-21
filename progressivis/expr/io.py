from .table import TableExpr, Expr, Pipeable
from progressivis.io.csv_loader import CSVLoader
from progressivis import Print


load_csv = Pipeable(TableExpr, CSVLoader)
echo = Pipeable(Expr, Print)
