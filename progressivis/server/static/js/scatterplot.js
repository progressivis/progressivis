var scatterplot_status = null;

function scatterplot_update(data) {
    module_status = data;
    module_run_number = data.last_update;
    module_update_table(data);
    scatterplot_update_vis(data);
}

function scatterplot_update_vis(data) {
}

function scatterplot_refresh() {
  module_get(scatterplot_update, module_error);
}

function scatterplot_socketmsg(message) {
    var txt = message.data, run_number;
    if (txt.startsWith("tick ")) {
	run_number = txt.substring(5);
	if (run_number > module_run_number)
	    scatterplot_refresh();
    }
    else 
	console.log('Module '+module_id+' received unexpected socket message: '+txt);
}


function scatterplot_ready() {
    scatterplot_refresh();
    websocket_open("module "+module_id, scatterplot_socketmsg);
}
