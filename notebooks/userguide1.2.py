# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Progressive Loading and Visualization
#
# This notebook shows the simplest code to download all the New York Yellow Taxi trips from 2015. They were all geolocated and the trip data is stored in multiple CSV files.
# We visualize progressively the pickup locations (where people have been picked up by the taxis).
#
# First, we define a few constants, where the file is located, the desired resolution, and the url of the taxi file.

# %%
import warnings
warnings.filterwarnings("ignore")
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
from progressivis import (CSVLoader, Histogram2D, ConstDict, Heatmap, PDict)

# Create a csv loader filtering out data outside NYC
csv = CSVLoader(LARGE_TAXI_FILE, usecols=['pickup_longitude', 'pickup_latitude'])
# Create a module to compute the min value progressively
min = ConstDict(PDict({'pickup_longitude': bounds.left, 'pickup_latitude': bounds.bottom}))
max = ConstDict(PDict({'pickup_longitude': bounds.right, 'pickup_latitude': bounds.top}))

# Create a module to compute the 2D histogram of the two columns specified
# with the given resolution
histogram2d = Histogram2D('pickup_longitude', 'pickup_latitude', xbins=RESOLUTION, ybins=RESOLUTION)
# Connect the module to the csv results and the min,max bounds to rescale
histogram2d.input.table = csv.output.result
histogram2d.input.min = min.output.result
histogram2d.input.max = max.output.result
# Create a module to create an heatmap image from the histogram2d
heatmap = Heatmap()
# Connect it to the histogram2d
heatmap.input.array = histogram2d.output.result

# %%
heatmap.display_notebook()
# Start the scheduler
csv.scheduler().task_start()

# %%
# Show what runs
csv.scheduler()

# %%
csv.scheduler().task_stop()

# %%
