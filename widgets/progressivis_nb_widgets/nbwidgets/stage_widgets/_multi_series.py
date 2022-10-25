multi_series_no_data = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "height": 400,
    "width": 400,
    "description": "Multi series template",
    "data": {"name": "data"},
    "mark": "line",
    "encoding": {
        "x": {"field": "date", "type": "temporal"},
        "y": {"field": "level", "type": "quantitative"},
        "color": {"field": "symbol", "type": "nominal"}
    }
}


_static_scatterplot_no_data = {
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "description": "A scatterplot template",
  "data":  {"name": "data"},
  "mark": "point",
  "encoding": {
      "x": {"field": "xcol", "type": "quantitative"},
      "y": {"field": "ycol", "type": "quantitative"},
      "color": {"field": "tcol", "type": "nominal"},
  }
}

scatterplot_no_data = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "height": 400,
    "width": 400,
    "description": "A scatterplot template",
    "data":  {"name": "data"},
    "mark": "point",
    "encoding": {
    }
}
