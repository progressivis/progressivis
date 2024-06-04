# Visualizations

There is a set of visualizations intended for Progressivis and usable with [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) grouped in a separate package called `ipyprogressivis`.

## Chaining widgets

As their name suggests, chaining widgets (CW) are graphical components based on Jupyter widgets that can be composed to implement data analysis scenarios. Their interconnection capabilities enable the creation of directed acyclic graphs (DAGs)

Each CW is designed for a specific stage of an analysis scenario (data loading, filtering, joins, etc.) and is associated with a sub-graph of PV modules in the background, usually grouped behind a front panel.

## Chaining widgets persistent settings

CWs require a file tree located in the user's homedir with the following structure:

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

### Data loaders

#### CSV loader

![](viz_images/csv_loader_cw.png)

* The `Bookmarks` field displays the contents (previously filled in by hand) of the `bookmarks` file in `$HOME/.progressivis`. Lines selected here represent urls ans docal files to be loaded. You can select one or more lines in this field. You can also ignore it and use the following field:
* `New URLs`: if the urls or local files present in bookmarks are not suitable, you can enter new paths here
* `URL to sniff`: Unique  url or local file to be used by the sniffer to discover data. If empty, the sniffer uses the first line among those selected for loading
* `Rows`: number of rows to be taken into account  by the sniffer to discover data
* `Throttle:`force the loader to limit the number of lines loaded at each step
* `Sniff ...` button: display the sniffer:

![](viz_images/sniffer.png)

Among other things, it allows you to customize parsing options, select the desired subset of columns and type them.

Once the configuration is complete, you can save it for later use and start loading.

![](viz_images/start_save_csv.png)

If “freeze” is checked, the configuration chosen here will be replayed as is, without any interaction when the current scenario is reused.

Once loading has begun, the `Next stage` list and the `Chain it` button will be used to attach a new CW to the treatment.

![](viz_images/next_stage.png)

#### PARQUET Loader

### Table operators
#### Group By

Groups the indexes of rows containing the same value for the selected column:

![](viz_images/group_by.png)

Given that tables can contain multi-dimensional values (in particular, the datetime type is represented as a vector with 6 elements: year, month, day, hour, minute, second), this CW introduces the notion of sub-columns, enabling rows to be grouped according to a subset of positions (6 sub-columns, in a datetime column). For example, indexes corresponding to the same day can be grouped together in a datetime column by selecting the first 3 sub-columns: year, month, day:

![](viz_images/group_by_datetime.png)

#### Aggregate

Allows predefined operations to be performed on table rows previously grouped via a GroupBy CW:

![](viz_images/aggregate.png)



#### Join
#### Virtual columnn creator (columns.py)
#### Facade creator

### Display (leaf) widgets
#### Dump table
#### Descriptive statistics (desc_stat.py)
#### Heatmap
#### Multi-series
#### Scatterplot
* ...

## Create a scenario

Notebooks hosting a scenario need to be initialized in a particular way. One call them **ProgressiBooks** and they must be created via the `Progressivis/New ProgressiBook` menu.

Once created, a `Run ProgressiVis` button will appear in the first `ProgressiBook` cell.

By clicking this button, the `ProgressiVis` scheduler will be launched and the start `CW` will appear. Currently, it proposes two data loading actions (CSV files and Parquetfiles), and offers the option (by checking the `record this scenario` box) of saving the session for later use.

![](viz_images/constructor_cw.png)
