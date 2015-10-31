var module_id = null;

function module_get(success, error) {
    $.post($SCRIPT_ROOT+'/progressivis/module/'+module_id)
        .done(success)
        .fail(error);
};

function module_update(data) {
    progressivis_update(data);
    module_update_table(data);
}

function module_update_table(data) {
    $('#module').html(layout_dict(data,
                                  ["classname",
                                   "state",
                                   "last_update",
                                   "default_step_size",
                                   "start_time",
                                   "end_time",
                                   "parameters",
                                   "input_slots",
                                   "output_slots"]));
    return;
    var columns = ['column', 'value'],
        vals = Object.keys(data).map(function(k) {
            return {column: k, value: data[k]};
        });
    
    var tr = d3
            .select("tbody")
            .selectAll("tr")
            .data(vals, function(d) { return d.column; });

    // Enter function
    var rows = tr.enter()
            .append("tr")
            .attr("class", "module-property");

    tr.order();

    var cells = rows.selectAll("td")
            .data(function(row) {
                return columns.map(function(column) {
                    return {column: column, value: row[column]};
                });
            })
            .enter()
            .append("td")
            .text(function(d) { return d.value; });

    // Update function
    tr.selectAll("td")
        .data(function(row) {
            return columns.map(function(column) {
                return {column: column, value: row[column]};
            });
        })
        .text(function(d) { return d.value; });

    // Exit function
    tr.exit().remove();
}

function module_refresh() {
  module_get(module_update, error);
}

function module_ready() {
    if (refresh == null) {
        refresh = module_refresh;
    }
    progressivis_ready("module "+module_id);
}
