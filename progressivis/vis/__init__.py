__all__ = ["Heatmap",
           "MCScatterPlot",
           "Histograms",
           "StatsExtender",
           "StatsFactory",
           "Histogram1dPattern",
           "DataShape"]


from progressivis.vis.heatmap import Heatmap

# from progressivis.vis.scatterplot import ScatterPlot
from progressivis.vis.histograms import Histograms
from .mcscatterplot import MCScatterPlot
from .stats_extender import StatsExtender
from .stats_factory import StatsFactory, DataShape, Histogram1dPattern
