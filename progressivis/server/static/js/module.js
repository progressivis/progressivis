var module_status = null,
    module_id = null,
    module_run_number = -1;

function module_get(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/module/'+module_id,
	dataType: 'json',
	method: 'POST'
    })
	.done(success)
	.fail(error);
};

function module_update(data) {
    module_status = data;
    module_run_number = data.last_update;
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
				   //"parameters",
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

function module_error(ev) {
  var contents = '<div class="alert alert-danger" role="alert">Error</div>';
  $('#error').html(contents);
}

function module_refresh() {
  module_get(module_update, module_error);
}

function module_socketmsg(message) {
    var txt = message.data, run_number;
    if (txt.startsWith("tick ")) {
	run_number = txt.substring(5);
	if (run_number > module_run_number)
	    module_refresh();
    }
    else 
	console.log('Module '+module_id+' received unexpected socket message: '+txt);
}


function module_ready() {
    module_refresh();
    websocket_open("module "+module_id, module_socketmsg);
}
