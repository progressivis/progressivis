# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     formats: ipynb,py:percent
#     rst2md: false
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Progressive Loading and Visualization
#
# This notebook shows the simplest code to download and visualize all the New York Yellow Taxi trips from January 2015.
# The trip data is stored in multiple CSV files, containing geolocated taxi trips.
# We visualize progressively the pickup locations (where people have been picked up by the taxis).

# %%
# We make sure the libraries are reloaded when modified, and avoid warning messages
# %load_ext autoreload
# %autoreload 2
import warnings
warnings.filterwarnings("ignore")

# %%
# Some constants we'll need: the data file to download and final image size
LARGE_TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"
RESOLUTION=512

# %% [markdown]
# ## Create Modules
# First, create the four modules we need.

# %%
from progressivis import CSVLoader, Histogram2D, Quantiles, Heatmap

# Create a CSVLoader module, a Quantile module, a Histogram2D module, and a Heatmap module.

# The CSV Loader only loads two columns of interest here with the 'usecols' keyword.
csv = CSVLoader(LARGE_TAXI_FILE, usecols=['pickup_longitude', 'pickup_latitude'])
quantiles = Quantiles()
# This Histogram2D column will compute a 2D histogram from the 2 columns with a resolution
histogram2d = Histogram2D('pickup_longitude', 'pickup_latitude', xbins=RESOLUTION, ybins=RESOLUTION)
heatmap = Heatmap()

# %% [markdown]
# ## Connect Modules
#
# Then, connect the modules.

# %%
# Now, connect the modules to create the Dataflow graph
# Quantiles inputs a table and outputs the quantiles of all the numeric columns
quantiles.input.table = csv.output.result

# Histogram inputs a table, the minimum values for the columns, and the maximum values.
histogram2d.input.table = csv.output.result
histogram2d.input.min = quantiles.output.result[0.03]  # 0.03 quantile
histogram2d.input.max = quantiles.output.result[0.97]  # 0.97 quantile
heatmap.input.array = histogram2d.output.result

# %% [markdown]
# ## Display the Heatmap

# %%
heatmap.display_notebook()

# %% [markdown]
# ## Start the scheduler

# %%
csv.scheduler.task_start()

# %% [markdown]
# ## Show the modules
# printing the scheduler shows all the modules and their states

# %%
csv.scheduler

# %% [markdown]
# ## Stop the scheduler
# To stop the scheduler, uncomment the next cell and run it

# %%

# csv.scheduler.task_stop()
