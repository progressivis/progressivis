hist1d_spec_no_data = {
    "data": {
        "name": "data"
    },
    "height": 500,
    "width": 500,
    "layer": [
        {
            "mark": "bar",
            "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
            "encoding": {
                "x": {"type": "ordinal",
                      "field": "nbins",
                      #"title": "Values", #"axis": {"format": ".2e", "ticks": False},
                      "title": "Values",
                      "axis": {"format": ".2e", "labelExpr": "(datum.value%10>0 ? null : datum.value)"},
                      #"axis": {"labelExpr": "datum.label"},
                },
                "y": {"type": "quantitative", "field": "level", "title": "Count"},

            }
        },
        {
            "mark": "rule",
            "encoding": {
                "x": {"aggregate": "min", "field": "bins", "title": None, "axis": {"tickCount": 0}},
                "color": {"value": "red"},
                "size": {"value": 1}
            }
        },
        {
            "mark": "rule",
            "encoding": {
                "x": {"aggregate": "max", "field": "bins", "title": None, "axis": {"tickCount": 0}},
                "color": {"value": "red"},
                "size": {"value": 1}
            }
        }

    ]
}
