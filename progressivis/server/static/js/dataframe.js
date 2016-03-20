var module_id = null,
    dataframe_slot = null;

function dataframe_get(success, error) {
    $.post($SCRIPT_ROOT+'/progressivis/module/df/'+module_id+'/'+dataframe_slot)
        .done(success)
        .fail(error);
}

function dataframe_update(data) {
    progressivis_update(data);
    dataframe_update_table(data);
}

function dataframe_update_table(data) {
    var columns = data['columns'].map(function(c) { return {"sTitle": c.toString()}; }),
        dataSet = data['data'],
        index = data['index'];
    for (var i = 0; i < index.length; i++) {
        dataSet[i].unshift(index[i].toString());
    }
    columns.unshift({"sTitle": 'index'});

    $('#dataframe').DataTable( {
        "data": dataSet,
        "columns": columns,
        "deferRender": true
    });
}

function dataframe_refresh() {
  dataframe_get(dataframe_update, error);
}

function dataframe_ready() {
    if (refresh == null) {
        refresh = dataframe_refresh;
    }
    progressivis_ready("dataframe "+module_id);
}
