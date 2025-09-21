LARGE_TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"
RESOLUTION=512


# %%
# See https://en.wikipedia.org/wiki/Module:Location_map/data/USA_New_York_City
from dataclasses import dataclass
@dataclass
class Bounds:
    top: float = 40.92
    bottom: float = 40.49
    left: float = -74.27
    right: float = -73.68

bounds = Bounds()


# %%
from progressivis import (
    CSVLoader, Histogram2D, ConstDict, Heatmap, PDict,
    BinningIndexND, RangeQuery2D, Variable
)
import progressivis.core.aio as aio

col_x = "pickup_longitude"
col_y = "pickup_latitude"
bnds_min = PDict({col_x: bounds.left, col_y: bounds.bottom})
bnds_max = PDict({col_x: bounds.right, col_y: bounds.top})
# Create a csv loader for the taxi data file
csv = CSVLoader(LARGE_TAXI_FILE, usecols=[col_x, col_y])
# Create an indexing module on the csv loader output columns
index = BinningIndexND()
# Creates one index per numeric column
index.input.table = csv.output.result[col_x, col_y]
# Create a querying module
query = RangeQuery2D(column_x=col_x,
                     column_y=col_y)
# Variable modules allow to dynamically modify queries ranges
var_min = Variable(name="var_min")
var_max = Variable(name="var_max")
query.input.lower = var_min.output.result
query.input.upper = var_max.output.result
query.input.index = index.output.result
query.input.min = index.output.min_out
query.input.max = index.output.max_out
# Create a module to compute the 2D histogram of the two columns specified
# with the given resolution
histogram2d = Histogram2D(col_x, col_y, xbins=RESOLUTION, ybins=RESOLUTION)
# Connect the module to the csv results and the min,max bounds to rescale
histogram2d.input.table = query.output.result
histogram2d.input.min = query.output.min
histogram2d.input.max = query.output.max
# Create a module to create an heatmap image from the histogram2d
heatmap = Heatmap()
# Connect it to the histogram2d
heatmap.input.array = histogram2d.output.result

scheduler = csv.scheduler
