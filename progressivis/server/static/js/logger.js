function logger () {
    var firstTime = true;

    function logger_get(success, error) {
        progressivis_get('/progressivis/logger', success, error);
    };

    function logger_update(data) {
        progressivis_update(data);
        firstTime = false;
        logger_update_table(data);
    }

    function table_row(entry) {

    }

    function logger_update_table(data) {
        var loggers = data.loggers,
            prefixes = ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'];
        for (var i = 0; i < loggers.length; i++) {
            var level = loggers[i].level;
        }
                        // var columns = ['module', 'level'];
                        // var tr = d3
                        //         .select("tbody")
                        //         .selectAll("tr")
                        //         .data(data.loggers, function(d) { return d.module; });

                        // // Enter function
                        // var rows = tr.enter()
                        //         .append("tr")
                        //         .attr("class", "loggers")
                        //         .on("click", logger_filter_module);

                        // tr.order();

                        // var cells = rows.selectAll("td")
                        //         .data(function(row) {
                        //             return columns.map(function(column) {
                        //                 return {column: column, value: row[column]};
                        //             });
                        //         })
                        //         .enter()
                        //         .append("td")
                        //         .text(function(d) { return d.value; });

                        // // Update function
                        // tr.selectAll("td")
                        //     .data(function(row) {
                        //         return columns.map(function(column) {
                        //             return {column: column, value: row[column]};
                        //         });
                        //     })
                        //     .text(function(d) { return d.value; });

                        // // Exit function
                        // tr.exit().remove();
                       }

    function logger_refresh(json) {
        logger_get(logger_update, error);
    }

    function logger_filter_module(module) {
        var url = $SCRIPT_ROOT+'/progressivis/logger/set_level/'+module.id;
    }

    function logger_ready() {
        refresh = logger_refresh;
        progressivis_ready("logger");
    }

    return logger_ready;
}

var logger_ready = logger();

