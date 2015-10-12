var scatterplot_status = null;


function scatterplot_ready() {
    module_refresh();
    websocket_open("module "+module_id, module_socketmsg);
}
