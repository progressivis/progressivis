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

# %%
from progressivis import Max, Print, RandomPTable


# %%
def _terse(_):
    print(".", end="", flush=True)


# %%
random = RandomPTable(10, rows=10_000)
# produces 10 columns named _1, _2, ...
max_ = Max()
max_.input[0] = random.output.result["_1", "_2", "_3"]
# Slot hints to restrict columns to ("_1", "_2", "_3")
pr = Print(proc=_terse)
pr.input[0] = max_.output.result

# %% [markdown]
# # Visualize the Dataflow Graph
# The Dataflow graph of a program can be visualized as follows, provided that the "graphviz" python package is installed, and that the "graphviz" program is installed as well.
# On a Linux and MacOS system with anaconda installed, you can install them with:
# ```
# conda install graphviz
# pip install graphviz
# ```

# %%
try:
    import graphviz
    src = random.scheduler.to_graphviz()
    gvz = graphviz.Source(src)
    display(gvz)
except Exception as e:
    print("Exception trying to visualize the Dataflow graph:", e)
    pass

# %% [markdown]
# # Running a ProgressiVis Program
# If the program is not using any graphical output, it can be run as a standard python program:
# ```
# python Summary.py
# ```

# %%
if __name__ != '__main__':  # run outside a notbebook
    random.scheduler.task_start()
else:
    from progressivis.core import aio
    aio.run(random.scheduler.start())
