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
import progressivis as pv
from progressivis.io import CSVLoader
from progressivis.stats import Histogram2D, Min, Max
from progressivis.vis import Heatmap

# Function to filter out trips outside of NYC.
# Since there are outliers in the files.
def filter_(df):
    lon = df['pickup_longitude']
    lat = df['pickup_latitude']
    #return df[(lon>-74.10)&(lon<-73.7)&(lat>40.60)&(lat<41)]
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

# %% [markdown]
# # Visualize the results
#
# We use barebone JupyterLab widgets to visualize the results.
# We will visualize an image each time the heatmap module is updated.

# %%
import ipywidgets as ipw
from IPython.display import display

# Create an ipywidget showing an image. It will be update when
# the heatmap is updated
wg = ipw.Image(value=b'\x00', width=RESOLUTION, height=RESOLUTION)
# Create a textbox to show the run number (akin to an epoch in ml)
wint = ipw.IntText(value=0, disabled=True)
# Create a button to stop the progressive program
bstop = ipw.Button(description="Stop")

# Stopping boils down to calling Scheduler.task_stop on the right scheduler
def stop(b):
    csv.scheduler().task_stop()
bstop.on_click(stop)

display(wg, ipw.HBox([wint, bstop]))

# Callback to update the image
async def _after_run(m, run_number):
    global wg, wint
    img = m.get_image_bin()  # get the image from the heatmap
    if img is None:
        return
    wg.value = img  # Replace the displayed image with this new one
    wint.value = m.last_update()  # also show the run number

# Install the callback
heatmap.on_after_run(_after_run)

# The image is shown below, and will be updated in place

# %%
# Start the scheduler
csv.scheduler().task_start()

# %%
