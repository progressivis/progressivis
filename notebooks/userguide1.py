# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     rst2md: false
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
# First, we define a few constants, where the file is located, the desired resolution, and the bounds of New York City.

# %%
LARGE_TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"
RESOLUTION=512

# See https://en.wikipedia.org/wiki/Module:Location_map/data/USA_New_York_City
bounds = {
	"top": 40.92,
	"bottom": 40.49,
	"left": -74.27,
	"right": -73.68,
}

# %%
from progressivis.io import CSVLoader
from progressivis.stats import Histogram2D, Min, Max
from progressivis.vis import Heatmap

# Function to filter out trips outside of NYC.
# Since there are outliers in the files.
def filter_(df):
    lon = df['pickup_longitude']
    lat = df['pickup_latitude']
    return df[
        (lon>bounds["left"]) &
        (lon<bounds["right"]) &
        (lat>bounds["bottom"]) &
        (lat<bounds["top"])
    ]

# Create a csv loader filtering out data outside NYC
csv = CSVLoader(LARGE_TAXI_FILE, index_col=False, filter_=filter_)

# Create a module to compute the min value progressively
min = Min()
# Connect it to the output of the csv module
min.input.table = csv.output.result
# Create a module to compute the max value progressively
max = Max()
# Connect it to the output of the csv module
max.input.table = csv.output.result
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

# %%
# Start the scheduler
csv.scheduler().task_start()

# %%
csv.scheduler()

# %%
csv.scheduler().task_stop()


# %%
