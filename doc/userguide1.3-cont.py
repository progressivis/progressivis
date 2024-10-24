import ipywidgets as widgets

# Yield control to the scheduler to start
await aio.sleep(1)
# Assign an initial value to the min and max variables
await var_min.from_input(bnds_min)
await var_max.from_input(bnds_max);

long_slider = widgets.FloatRangeSlider(
    value=[bnds_min[col_x], bnds_max[col_x]],
    min=bnds_min[col_x],
    max=bnds_max[col_x],
    step=(bnds_max[col_x]-bnds_min[col_x])/10,
    description='Longitude:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
lat_slider = widgets.FloatRangeSlider(
    value=[bnds_min[col_y], bnds_max[col_y]],
    min=bnds_min[col_y],
    max=bnds_max[col_y],
    step=(bnds_max[col_y]-bnds_min[col_y])/10,
    description='Latitude:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
def observer(_):
    async def _coro():
        long_min, long_max = long_slider.value
        lat_min, lat_max = lat_slider.value
        await var_min.from_input({col_x: long_min, col_y: lat_min})
        await var_up.from_input({col_x: long_max, col_y: lat_max})
    aio.create_task(_coro())
long_slider.observe(observer, "value")
lat_slider.observe(observer, "value")
widgets.VBox([long_slider, lat_slider])
