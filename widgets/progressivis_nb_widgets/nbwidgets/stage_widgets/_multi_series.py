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
