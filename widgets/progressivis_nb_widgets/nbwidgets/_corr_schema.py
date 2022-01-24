# https://stackoverflow.com/questions/65015676/vega-lite-x-axis-labels-not-showing-in-full
corr_spec_no_data = {
    "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
    "config": {"view": {"continuousHeight": 300, "continuousWidth": 400}},
    "data": {"name": "data"},
    "layer": [
        {
            "encoding": {
                "color": {"field": "corr", "type": "quantitative"},
                "x": {"field": "var2", "type": "ordinal"},
                "y": {"field": "var", "type": "ordinal"},
            },
            "height": 400,
            "mark": "rect",
            "width": 400,
        },
        {
            "encoding": {
                "color": {
                    "condition": {"test": "(datum.corr > 0.5)", "value": "white"},
                    "value": "black",
                },
                "text": {"field": "corr_label", "type": "nominal"},
                "x": {
                    "field": "var2",
                    "type": "ordinal",
                    "axis": {"labelAngle": -30, "labelLimit": 10000},
                },
                "y": {"field": "var", "type": "ordinal"},
            },
            "height": 400,
            "mark": "text",
            "width": 400,
        },
    ],
}
