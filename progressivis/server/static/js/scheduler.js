var scheduler_status = null,
    scheduler_run_number = -1;

function scheduler_get(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler/',
	dataType: 'json',
	method: 'POST'
    })
	.done(success)
	.fail(error);
};

function scheduler_start(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler/start',
	dataType: 'json',
	method: 'POST'
    })
	.done(success)
	.fail(error);
};

function scheduler_stop(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler/stop',
	dataType: 'json',
	method: 'POST'
    })
	.done(success)
	.fail(error);
};

function cmp_order(mod1, mod2) {
  if (mod1.order == mod2.order) return 0;
  if (mod1.order == undefined) return -1;
  return 1;
}

function scheduler_update(data) {
    var module, i;

    //console.log('scheduler update');

    scheduler_status = data;
    scheduler_run_number = data['run_number'];
    if (data.is_running) {
	$('#start').addClass('disabled');
	$('#stop').removeClass('disabled');
    }
    else if (data.is_terminated) {
	$('#start').removeClass('disabled');
	$('#stop').removeClass('disabled');
    }
    else {
	$('#start').removeClass('disabled');
	$('#stop').addClass('disabled');
    }
    scheduler_update_table(data);
}

function scheduler_update_table(data) {
    var columns = ['id', 'classname', 'state', 'last_update', 'order'];
    var tr = d3
	    .select("tbody")
	    .selectAll("tr")
	    .data(data.modules, function(d) { return d.id; });

    // Enter function
    var rows = tr.enter()
	    .append("tr")
	    .attr("class", "module")
	    .on("click", scheduler_show_module);

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

function scheduler_error(ev) {
  var contents = '<div class="alert alert-danger" role="alert">Error</div>';
  $('#error').html(contents);
}

function scheduler_refresh() {
  scheduler_get(scheduler_update, scheduler_error);
}

function scheduler_show_module(module) {
    var url = $SCRIPT_ROOT+'/progressivis/module/'+module.id;
    var win = window.open(url, '_blank');
    win.focus();
}

function scheduler_socketmsg(message) {
    var txt = message.data, run_number;
    if (txt.startsWith("tick ")) {
	run_number = Number(txt.substr(5));
	if (run_number > scheduler_run_number)
	    scheduler_refresh();
    }
    else 
	console.log('Scheduler received unexpected socket message: '+txt);
}

function scheduler_ready() {
    $('#start').click(function() { scheduler_start(scheduler_refresh, scheduler_error); });
    $('#stop').click(function() { scheduler_stop(scheduler_refresh, scheduler_error); });
    scheduler_refresh();
    $('#modules-tab a').click(function (e) {
	e.preventDefault();
	$(this).tab('show');
    });
    websocket_open("scheduler", scheduler_socketmsg);
}
