# Visualizations

There is a set of visualizations intended for Progressivis and usable with [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) grouped in a separate package called `ipyprogressivis`.

## Create a scenario

Notebooks hosting a progressive scenario need to be initialized in a particular way. One call them **ProgressiBooks** and they must be created via the `Progressivis/New ProgressiBook` menu.

Once created, a `Run ProgressiVis` button will appear in the first `ProgressiBook` cell.

By clicking this button, the `ProgressiVis` scheduler will be launched and the start box will appear. Currently, it proposes two data loading actions (CSV files and PARQUET files), and offers the option (by checking the `record this scenario` box) of saving the session for later use. These actions, like all other actions in the scenario, are implemented via a set of `chaining widgets`.

![](viz_images/constructor_cw.png)


## Chaining widgets

As their name suggests, chaining widgets (`CW`) are graphical components based on Jupyter widgets that can be composed to implement data analysis scenarios. Their interconnection capabilities enable the creation of directed acyclic graphs (DAGs)

Each `CW` is designed for a specific stage of an analysis scenario (data loading, filtering, joins, etc.) and is associated with a sub-graph of PV modules in the background, usually grouped behind a front panel.

## Chaining widgets persistent settings

`CW`s require a file tree located in the user's homedir with the following structure:

```
.progressivis/
├── bookmarks
└── widget_settings
    └── CsvLoaderW
        ├── taxis
        └── weather
    └── ParquetLoaderW
        ├── iris
        └── penguins
    ...
    ...
```

## Chaining widgets list

### Data loaders category

#### CSV loader

Possible topology:

![](viz_images/csv_loader_topology.png)

##### Function:

It loads one or many CSV files progressively

After starting, the main interface is:

![](viz_images/csv_loader_cw.png)

Where:

* The `Bookmarks` field displays the contents (previously filled in by hand) of the `bookmarks` file in `$HOME/.progressivis`. Lines selected here represent urls ans docal files to be loaded. You can select one or more lines in this field. You can also ignore it and use the following field:
* `New URLs`: if the urls or local files present in bookmarks are not suitable, you can enter new paths here
* `URL to sniff`: Unique  url or local file to be used by the sniffer to discover data. If empty, the sniffer uses the first line among those selected for loading
* `Rows`: number of rows to be taken into account  by the sniffer to discover data
* `Throttle:`force the loader to limit the number of lines loaded at each step
* `Sniff ...` button: displays the sniffer (image below):

![](viz_images/sniffer.png)

The sniffer, among other things, allows you to customize parsing options, select the desired subset of columns and type them.

Once the configuration is complete, you can save it for later use and start loading.

![](viz_images/start_save_csv.png)

If “freeze” is checked, the configuration chosen here will be replayed as is, without any interaction when the current scenario is reused.

Once loading has begun, the `Next stage` list and the `Chain it` button will be used to attach a new `CW` to the treatment.

![](viz_images/next_stage.png)

#### PARQUET Loader

... comming soon

### Table operators category

#### Group By

Possible topology:

![](viz_images/group_by_topology.png)

##### Function:

It groups the indexes of rows containing the same value for the selected column:

![](viz_images/group_by.png)

Given that tables can contain multi-dimensional values (in particular, the datetime type is represented as a vector with 6 elements: year, month, day, hour, minute, second), this `CW` introduces the notion of sub-columns, enabling rows to be grouped according to a subset of positions (6 sub-columns, in a datetime column). For example, indexes corresponding to the same day can be grouped together in a datetime column by selecting the first 3 sub-columns: year, month, day:

![](viz_images/group_by_datetime.png)

#### Aggregate

Possible topology:

![](viz_images/aggregate_topology.png)

##### Function:

Allows predefined operations to be performed on table rows previously grouped via a **Group by** `CW`:

![](viz_images/aggregate.png)

Each input (column, operation) pair generates a dedicated column in the output table:

![](viz_images/aggregate_columns.png)



#### Join

Possible topology:

![](viz_images/join_topology.png)


##### Function:

Performs a join between two table outputs via one or more columns. Sub-column joins (in the sense described above for group-by) are also supported.

ProgressiVis currently supports `one to one` and `one to many` joins (but not `many to many`).

In a `one to many` join, the table on the `one` side is called `primary` and the table on the `many` side is called `related`.

Obviously, in a `one to one` join, the two roles are interchangeable:

The first step is to select the two inputs and define their respective roles then click `OK`:

![](viz_images/join.png)

We can now define the join and select the columns to be kept in the join from among those in the primary table:

![](viz_images/join_primary.png)

... and those in the related table:

![](viz_images/join_related.png)

The resultant join table is:

![](viz_images/join_result.png)


#### Virtual columnn creator (columns.py)
#### Facade creator

### Display (leaf) category
#### Dump table
#### Descriptive statistics (desc_stat.py)
#### Heatmap
#### Multi-series
#### Scatterplot
* ...
