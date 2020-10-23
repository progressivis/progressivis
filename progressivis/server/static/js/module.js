var module_id = null;

function module_get(success, error) {
    return progressivis_get('/progressivis/module/get', success, error, module_id);
}

function module_set_hotline(status, error) {
    if(status){
        return progressivis_get('/progressivis/module/hotline_on', function(_){}, error, module_id);
    } else {
        return progressivis_get('/progressivis/module/hotline_off', function(_){}, error, module_id);
    }
}

function module_input(data, success, error, module) {
    if (! module)
        module = module_id;
    if (handshake) {
        console.log("socketio input request "+data);
        if (data != null && typeof(data)!='string')
            data = JSON.stringify({path: module,
                                   var_values: data});
        return new Promise((resolve, reject) =>
                           socket.emit('/progressivis/module/input',
                                       data, resolve))
            .then(success)
            .catch(error);
    }
    else {
        if (data != null && typeof(data)!='string')
            data = JSON.stringify(data);
        $.ajax({
            type: 'POST',
            url: $SCRIPT_ROOT+'/progressivis/module/input/'+module,
            data: data,
            success: success,
            contentType: "application/json",
            dataType: 'json'
        })
            .fail(error);
    }
}

function module_update(data, with_table=true) {
    progressivis_update(data);
    if(with_table){
        module_update_table(data);
    }

}

function module_show_dataframe(slot) {
    var url = $SCRIPT_ROOT+'/progressivis/module/df/'+module_id+'/'+slot;
    var win = window.open(url, '_blank');
    win.focus();
}

function speed_disp(v){
    var SIZE = 15;
    var last = v[v.length-1];
    last = last.toFixed(0);
    last = last+"&nbsp;".repeat(SIZE-last.length);
    return "<table><tr><td>"+last+"</td><td><span class='speedsparkline'>...</span></td></tr></table>";
}

function module_update_table(data) {
    var slots = data['output_slots'],
        buttons = '<div class="btn-group" role="group" aria-label="DataFrames">\n';
    slots['_params'] = true;
    for (var slot in slots) {
        buttons += '<button type="button" class="btn btn-default slot">'+slot+'</button>\n';
    }
    buttons += '</div>';
    data['output_slots'] = buttons;
    
    $('#module').html(layout_dict(data,
                                  ["classname",
                                   "speed",
                                   "output_slots",
                                   "debug",
                                   "state",
                                   "last_update",
                                   "default_step_size",
                                   "start_time",
                                   "end_time",
                                   "parameters",
                                   "input_slots"], value_func={speed: speed_disp}));
    $('.speedsparkline').sparkline(data['speed']); 
    progressivis_get('/progressivis/module/quality', line_graph, error, module_id);
    $('.btn.slot').click(function() { module_show_dataframe($(this).text()); });
}

function module_refresh(json) {
    module_get(module_update, error);
}

function module_ready() {
    if (refresh == null) {
        refresh = module_refresh;
    }
    progressivis_ready("module "+module_id);
}


