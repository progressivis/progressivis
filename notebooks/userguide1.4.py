# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     formats: ipynb,py:percent
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
# This notebook shows a simple code to download and visualize all the New York Yellow Taxi trips from January 2015, knowing the bounds of NYC.
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
# ## Define NYC Bounds
# If we know the bounds, this will simplify the code.
# See https://en.wikipedia.org/wiki/Module:Location_map/data/USA_New_York_City

# %%
from dataclasses import dataclass
@dataclass
class Bounds:
    top: float = 40.92
    bottom: float = 40.49
    left: float = -74.27
    right: float = -73.68

bounds = Bounds()

# %% [markdown]
# ## Create Modules
# First, create the four modules we need.

# %%
from progressivis import CSVLoader, Histogram2D, ConstDict, Heatmap, PDict

# Create a CSVLoader module, two min/max constant modules, a Histogram2D module, and a Heatmap module.

csv = CSVLoader(LARGE_TAXI_FILE, usecols=['pickup_longitude', 'pickup_latitude'])
min = ConstDict(PDict({'pickup_longitude': bounds.left, 'pickup_latitude': bounds.bottom}))
max = ConstDict(PDict({'pickup_longitude': bounds.right, 'pickup_latitude': bounds.top}))
histogram2d = Histogram2D('pickup_longitude', 'pickup_latitude', xbins=RESOLUTION, ybins=RESOLUTION)
heatmap = Heatmap()

# %% [markdown]
# ## Connect Modules
#
# Then, connect the modules.

# %%
histogram2d.input.table = csv.output.result
histogram2d.input.min = min.output.result
histogram2d.input.max = max.output.result
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
# ## Module Quality
# Most modules performing a computation can return a "quality" measure.
# What is a "quality" is a long question, but for ProgressiVis, it is a floating point number;
# the higher, the better. A module's output can only be trusted if its quality is stable.
# Unfortunately, there are cases when the quality will remain stable for a while and change again,
# but we'll ignore them for now.
#
# For the `Min` module, the quality is simply the negative value of the columns. The higher, the better, and when they stabilize, the module becomes trustworthy. Here, `min` is a constant so it does not return a quality.
#
# The `Histogram2D` module has a more complex quality based on the difference between the array values between runs, 0 being best.
#

# %%
histogram2d.get_quality()

# %% [markdown]
# ## Module Progress
# Module can return their progress, a pair of two values: (current, maximum).
# Both can vary each time the module is run, since the maximum is usually an estimate.

# %%
min.get_progress()

# %% [markdown]
# ## Visualizing the Quality and Progress Bar
# We define two functions to monitor the quality and progress here.

# %%
import ipywidgets as ipw

from progressivis import Module
from ipyprogressivis.widgets import QualityVisualization

def display_quality(mods, period: float = 3) -> QualityVisualization:
    qv = QualityVisualization()
    last = 0  # show immediately
    if isinstance(mods, Module):
        mods = [mods]

    async def _after_run(m: Module, run_number: int) -> None:
        nonlocal last
        now = m.last_time()
        if (now - last) < period:
            return
        last = now
        measures = m.get_quality()
        if measures is not None:
            qv.update(measures, now)

    for mod in mods:
        mod.on_after_run(_after_run)
    return qv


def display_progress_bar(mod: Module, period: float = 3) -> ipw.IntProgress:
    prog_wg = ipw.IntProgress(
        description="Progress", min=0, max=1000, layout={"width": "200"}
    )

    def _proc(m: Module, r: int) -> None:
        val_, max_ = m.get_progress()
        prog_wg.value = val_
        if prog_wg.max != max_:
            prog_wg.max = max_
    mod.on_after_run(_proc)
    return prog_wg


# %% [markdown]
# ## Monitoring the Quality
#
# The quality can be visualized when the module runs, with a controlled updated every 3s to avoid flooding the notebook and the user.

# %%
heatq = display_quality(heatmap)
heatq

# %% [markdown]
# The quality widget can be manipulated dynamically to change its size according to, e.g., its level of interest.

# %%
heatq.width = "100%"
heatq.height = 100

# %%
display_progress_bar(heatmap)

# %% [markdown]
# ## Stop the scheduler
# To stop the scheduler, uncomment the next cell and run it

# %%

# csv.scheduler.task_stop()
