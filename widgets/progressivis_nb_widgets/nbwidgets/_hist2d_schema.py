hist2d_spec_no_data = {
    "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
    "width": 500,
    "height": 400,
    "data": {"name": "data"},
    "encoding": {
        "color": {
            "field": "z",
            "type": "quantitative",
            # "scale": {
            # "domain": [0,1]  # Remove if domain changes
            # }
        },
        "x": {"field": "x", "type": "ordinal"},
        "y": {"field": "y", "type": "ordinal"},
    },
    "mark": "rect",
    "config": {"axis": {"disable": True}},  # Change to False to see the ticks
}
