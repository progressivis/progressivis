from progressivis import CSVLoader, Histogram2D, Quantiles, Heatmap

LARGE_TAXI_FILE = ("https://www.aviz.fr/nyc-taxi/"
                   "yellow_tripdata_2015-01.csv.bz2")
RESOLUTION=512

csv = CSVLoader(LARGE_TAXI_FILE, index_col=False,
                usecols=['pickup_longitude', 'pickup_latitude'])

quantiles = Quantiles()
quantiles.input.table = csv.output.result

histogram2d = Histogram2D('pickup_longitude', 'pickup_latitude',
                          xbins=RESOLUTION, ybins=RESOLUTION)
histogram2d.input.table = quantiles.output.table
histogram2d.input.min = quantiles.output.result[0.03]
histogram2d.input.max = quantiles.output.result[0.97]

heatmap = Heatmap()
heatmap.input.array = histogram2d.output.result

# heatmap.display_notebook()
scheduler = csv.scheduler()  # .task_start()
