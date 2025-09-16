# Example Gallery

This page shows different scenarios implemented with `ipyprogressivis`

(taxis-precipitations-scatterplot)=
## NYC Taxis / Precipitations  scatterplot

The notebook shown below is downloadable [here](https://github.com/progressivis/ipyprogressivis/blob/main/notebooks/taxis_precipitations_scatterplot.ipynb).
You can display this example in a <a href="_static/taxis_precipitations_scatterplot.html" target=_blank>new tab</a>.

```{eval-rst}
.. raw:: html

   <iframe src="_static/taxis_precipitations_scatterplot.html" height="1000px" width="100%"></iframe>

```

(taxis-precipitations-line-chart)=
## NYC Taxis / Precipitations  line chart

The notebook shown below is downloadable [here](https://github.com/progressivis/ipyprogressivis/blob/main/notebooks/taxis_precipitations_line_chart.ipynb).
You can display this example in a <a href="_static/taxis_precipitations_line_chart.html" target=_blank>new tab</a>.

```{eval-rst}
.. raw:: html

   <iframe src="_static/taxis_precipitations_line_chart.html" height="1000px" width="100%"></iframe>

```

(trip-rain-corr)=
## NYC Taxis / Correlation between the number of courses per day and precipitations

The notebook shown below is downloadable [here](https://github.com/progressivis/ipyprogressivis/blob/main/notebooks/trip_rain_corr.ipynb).
You can display this example in a <a href="_static/trip_rain_corr.html" target=_blank>new tab</a>.

```{eval-rst}
.. raw:: html

   <iframe src="_static/trip_rain_corr.html" height="1000px" width="100%"></iframe>

```

(scenario-with_snippets)=
## A scenario using `progressivis` snippets


The notebook shown below is downloadable [here](https://github.com/progressivis/ipyprogressivis/blob/main/notebooks/scenario_using_snippets.ipynb).
You can display this example in a <a href="_static/scenario_using_snippets.html" target=_blank>new tab</a>.

```{eval-rst}
.. raw:: html

   <iframe src="_static/scenario_using_snippets.html" height="2000px" width="100%"></iframe>

```

(taxis-borough)=
## A scenario using `duckdb` in a snippet


The notebook shown below is downloadable [here](https://github.com/progressivis/ipyprogressivis/blob/main/notebooks/taxis_borough.ipynb).
You can display this example in a <a href="_static/taxis_borough.html" target=_blank>new tab</a>.

```{eval-rst}
.. note::
   This example requires the `duckdb` package which is not part of `progressivis` dependencies.
```

```{eval-rst}
.. raw:: html

   <iframe src="_static/taxis_borough.html" height="2000px" width="100%"></iframe>

```

(taxis-trips-density-map)=

## A multi-class density map for NYC Taxis data.

There are two classes represented here:

* **A**: represents the pickup points density
* **B**: represents the dropoff points density

Pickup/dropoff sample points are also represented as red/blue dots.


The notebook shown below is downloadable [here](https://github.com/progressivis/ipyprogressivis/blob/main/notebooks/taxis_trips_density_map.ipynb).
You can display this example in a <a href="_static/taxis_trips_density_map.html" target=_blank>new tab</a>.

```{eval-rst}
.. raw:: html

   <iframe src="_static/taxis_trips_density_map.html" height="2000px" width="100%"></iframe>

```

(mb-kmeans-cluster-s1)=

## Progressive KMeans


This implémentation is based on [scikit-learn mini-batch KMeans](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans)

The clustering dataset came from [P. Fänti and S. Sieranoja, K-means properties on six clustering benchmark datasets](https://cs.joensuu.fi/sipu/datasets/)

The notebook shown below is downloadable [here](https://github.com/progressivis/ipyprogressivis/blob/main/notebooks/mb_kmeans_cluster_s1.ipynb).
You can display this example in a <a href="_static/mb_kmeans_cluster_s1.html" target=_blank>new tab</a>.

```{eval-rst}
.. raw:: html

   <iframe src="_static/mb_kmeans_cluster_s1.html" height="2000px" width="100%"></iframe>

```

(mnist-tsne-2d)=

## TSNE 2D


The implementation used here is based on [PANENE](https://github.com/e-/PANENE), an algorithm for the k-nearest neighbor (KNN) problem and more specificaly on [this example](https://github.com/e-/PANENE/tree/master/examples/tsne/responsive_tsne).

The algorithm is described in the paper: J. Jo, J. Seo and J. -D. Fekete, "PANENE: A Progressive Algorithm for Indexing and Querying Approximate k-Nearest Neighbors," in IEEE Transactions on Visualization and Computer Graphics, vol. 26, no. 2, pp. 1347-1360, 1 Feb. 2020, [doi: 10.1109/TVCG.2018.2869149](https://ieeexplore.ieee.org/document/8462793).

The `PANENE` implementation for `python` is called `pynene`.

As `pynene` is not part of progressivis you have to install it before running this example.

In an environment containing `progressivis`, `ipyprogressivis` and a `C++` compiler you can install `pynene` via `pip` this way:

```sh
pip install git+https://github.com/progressivis/PANENE.git@progressivis
```
The notebook shown below is downloadable [here](https://github.com/progressivis/ipyprogressivis/blob/main/notebooks/mnist_tsne2d.ipynb).
You can display this example in a <a href="_static/mnist_tsne2d.html" target=_blank>new tab</a>.


```{eval-rst}
.. raw:: html

   <iframe src="_static/mnist_tsne2d.html" height="2000px" width="100%"></iframe>

```

