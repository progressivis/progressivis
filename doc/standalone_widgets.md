# Standalone widgets

As mentioned above, the main role of the ipyprogressivis package is to provide progressivis with the extensions it needs to run in progressive notebooks.

Nevertheless, it can be useful as a stand-alone package (without progressivis) to benefit even outside the progressive context from a number of widgets that are part of the progressive extensions mentioned above.

If you want to use these widgets without ProgressiVis, simply install ipyprogressivis as explained here.

This section explains how these widgets work.

## The Sniffer
![](viz_images/sniffer.png)

The **sniffer**, among other things, allows you to customize parsing options, select the desired subset of columns and type them.

Once the configuration is complete, you can save it for later use, so you don't have to refill all the options manually, and start loading.

TODO


## The DAG Widget

The **DAG Widget** visualizes a directed acyclic graph of notebook cell dependencies. It can be used to navigate between dependent cells, and also to monitor the status of these cells.
It is used by ipyprogressivis to show the graph of Chained Widgets and the status of these widgets.

TODO

