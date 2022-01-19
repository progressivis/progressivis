hist1d_spec_no_data = {
    "data": {"name": "data"},
    "height": 500,
    "width": 800,
    "layer": [
        {
            "mark": "bar",
            "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
            "encoding": {
                "x": {
                    "type": "ordinal",
                    "field": "nbins",
                    "title": "Values",
                    "axis": {
                        "labelExpr": "(datum.value%10>0 ? null : "
                        "format(data('data')[datum.value].xvals, '.2f'))"
                    },
                },
                "y": {"type": "quantitative", "field": "level", "title": "Histogram"},
            },
        }
    ],
}

kll_spec_no_data = {
    "data": {"name": "data"},
    "height": 500,
    "width": 800,
    "layer": [
        {
            "mark": "bar",
            "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
            "encoding": {
                "x": {
                    "type": "ordinal",
                    "field": "nbins",
                    "title": "Values",
                    "axis": {
                        "labelExpr": "(datum.value%10>0 ? null : "
                        "format(data('data')[datum.value].xvals, '.2f'))"
                    },
                },
                "y": {"type": "quantitative", "field": "level", "title": "Sketching"},
            },
        },
        {
            "mark": "rule",
            "encoding": {
                "x": {"type": "ordinal", "aggregate": "max", "field": "rule_lower"},
                "color": {"value": "red"},
                "size": {"value": 3},
            },
        },
        {
            "mark": "rule",
            "encoding": {
                "x": {"type": "ordinal", "aggregate": "max", "field": "rule_upper"},
                "color": {"value": "red"},
                "size": {"value": 3},
            },
        },
    ],
}
