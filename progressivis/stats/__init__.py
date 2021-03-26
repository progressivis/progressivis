
from .stats import Stats
from .min import Min, ScalarMin
from .max import Max, ScalarMax
from .idxmax import IdxMax
from .idxmin import IdxMin
from .var import Var, VarH
from .percentiles import Percentiles
#from .linear_regression import LinearRegression
from .histogram1d import Histogram1D
from .histogram2d import Histogram2D
from .mchistogram2d import MCHistogram2D
from .sample import Sample
from .random_table import RandomTable, RandomDict
#from .kernel_density import KernelDensity


__all__ = ["Stats",
           "Min",
           "Max",
           "IdxMax",
           "IdxMin",
           "VarH",
           "Var",
           "Percentiles",
#           "LinearRegression",
           "Histogram1D",
           "Histogram2D",
           "MCHistogram2D",           
           "Sample",
           "RandomTable",
           "RandomDict"
               #"KernelDensity"
               ]

