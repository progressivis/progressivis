var firstTime = true;

var current_xhr = null;

function clear_current_xhr(){
    current_xhr = null;
}

function scheduler_get(success, error) {
    if(current_xhr){
        return;
    }
    console.log('Ajax query for scheduler');    
    current_xhr = $.post($SCRIPT_ROOT+'/progressivis/scheduler/?short=False')
        .done(success)
        .fail(error)
        .always(clear_current_xhr);
};

function cmp_order(mod1, mod2) {
  if (mod1.order == mod2.order) return 0;
  if (mod1.order == undefined) return -1;
  return 1;
}

function scheduler_update(data) {
    progressivis_update(data);
    module_graph.update_vis(data.modules, firstTime);
    firstTime = false;
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

function scheduler_refresh(json) {
    if (json && json.modules)
        scheduler_update(json);
    else
        scheduler_get(scheduler_update, error);
}

function scheduler_show_module(module) {
    var url = $SCRIPT_ROOT+'/progressivis/module/get/'+module.id;
    var win = window.open(url, '_blank');
    win.focus();
}

function scheduler_ready() {
    $('#modules-tab a').click(function (e) {
        e.preventDefault();
        $(this).tab('show');
    });
    module_graph.setup();
    refresh = scheduler_refresh;
    progressivis_ready("scheduler");
}
