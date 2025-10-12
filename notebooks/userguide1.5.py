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
# # Dynamic Modification of a ProgressiVis Program
#
# This notebook shows how a running ProgressiVis program can be modified.
# A simple program is created that loads a CSV file, compute the minumum, and prints a dot for each chunk.
# Then, the program is modified; a max module is added after the CSV module and a slash is printed at each chunk.
# Finally, the initial min module is deleted, triggering the deletion of the first print a **collateral dammage**.

# %%
from progressivis import (
    CSVLoader,
    Print,
    Min, Max,
)

# %%
import warnings
warnings.filterwarnings("ignore")
LARGE_TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"

# prints a dot at each chunk (run_step)
def terse(x):
    print(".", end="", flush=True)

# prints a slash at each chunk
def terse2(x):
    print("/", end="", flush=True)


# %% [markdown]
# ## Creates the initial program
#
# The print module calls the function `terse` at each chunk, showing a dot when run.

# %%
csv = CSVLoader(LARGE_TAXI_FILE, usecols=['pickup_longitude', 'pickup_latitude'])
m = Min(name="min")
prt = Print(proc=terse)
m.input.table = csv.output.result
prt.input.df = m.output.result

# %%
csv.scheduler.task_start()

# %%
csv.scheduler

# %% [markdown]
# # Adds a branch
#
# Compute the max and prints each chunk with a slash.
# The branch is validated at the end of the `with` construct and, if valid, is run. Otherwise, an exception is raised and the initial program is not modified.

# %%
with csv.scheduler as dataflow:
    M = Max(name="max")
    prt2 = Print(proc=terse2)
    M.input.table = csv.output.result
    prt2.input.df = M.output.result

# %%
csv.scheduler

# %% [markdown]
# # Removes the initial branch
#
# The min module is deleted, which also deletes the first print module.

# %%
with csv.scheduler as dataflow:
    deps = dataflow.collateral_damage("min")
    print("The collateral damage of deleting min is:", deps)
    dataflow.delete_modules(*deps)

# %%
csv.scheduler

# %%
# csv.scheduler.task_stop()
