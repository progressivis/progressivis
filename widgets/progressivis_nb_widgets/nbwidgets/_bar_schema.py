bar_spec_no_data = {
    "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
    "height": 400,
    "width": 400,
    "data": {"name": "data"},
    "mark": "bar",
    "encoding": {
        "x": {"field": "category", "type": "ordinal", "title": "Categories"},
        "y": {"type": "quantitative", "field": "count", "title": "Count"},
    },
}
