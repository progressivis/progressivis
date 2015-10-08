var scheduler_status = null;

function scheduler_get(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler/',
	dataType: 'json'
    })
	.done(success)
	.fail(error);
};

function scheduler_start(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler/start',
	dataType: 'json'
    })
	.done(success)
	.fail(error);
};

function scheduler_stop(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler/stop',
	dataType: 'json'
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

    console.log('scheduler update');

    scheduler_status = data;
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
    var columns = ['id', 'classname', 'state', 'last_update', 'order'];
    var tr = d3
	    .select("tbody")
	    .selectAll("tr")
	    .data(data.modules, function(d) { return d.id; });
    var rows = tr.enter()
	    .append("tr");

    var cells = rows.selectAll("td")
	    .data(function(row) {
		return columns.map(function(column) {
		    return {column: column, value: row[column]};
		});
	    })
	    .enter()
	    .append("td")
	    .text(function(d) { return d.value; });

    tr.selectAll("td")
	.data(function(row) {
	    return columns.map(function(column) {
		return {column: column, value: row[column]};
	    });
	})
	.text(function(d) { return d.value; });


    tr.exit().remove();

    $('#module-tabs-contents .dynamic').remove();
    for (i = 0; i <  data.modules.length; i++) {
	module = data.modules[i];
	
    }
    
}

function scheduler_error(ev) {
  contents = '<div class="alert alert-danger" role="alert">Error</div>';
  $('#scheduler').html(contents);
}

function scheduler_refresh() {
  scheduler_get(scheduler_update, scheduler_error);
}

function scheduler_add_tab(name) {
    console.log('add tab '+name);
}

function scheduler_remove_tab(name) {
    console.log('remove tab '+name);
}

function scheduler_ready() {
    $('#start').click(function() { scheduler_start(scheduler_refresh, scheduler_error); });
    $('#stop').click(function() { scheduler_stop(scheduler_refresh, scheduler_error); });
    scheduler_refresh();
    $('#modules-tab a').click(function (e) {
	e.preventDefault();
	$(this).tab('show');
    });
}
