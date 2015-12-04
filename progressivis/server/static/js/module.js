var module_id = null;

function module_get(success, error) {
    $.post($SCRIPT_ROOT+'/progressivis/module/get/'+module_id)
        .done(success)
        .fail(error);
}

function module_input(data, success, error, module) {
    if (! module)
        module = module_id;
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
};

function module_update(data) {
    progressivis_update(data);
    module_update_table(data);
}

function module_show_dataframe(slot) {
    var url = $SCRIPT_ROOT+'/progressivis/module/df/'+module_id+'/'+slot;
    var win = window.open(url, '_blank');
    win.focus();
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
                                   "output_slots",
                                   "debug",
                                   "state",
                                   "last_update",
                                   "default_step_size",
                                   "start_time",
                                   "end_time",
                                   "parameters",
                                   "input_slots"]));

    $('.btn.slot').click(function() { module_show_dataframe($(this).text()); });
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
