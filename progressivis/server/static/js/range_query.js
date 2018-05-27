var range_query = function() {
    var sliders = {};

    function ready(){
        refresh = range_query_refresh;
        module_ready();
    }
    
    function range_query_refresh(){
        module_get(update, error);
    }

    function range_query_input(evt, value, d){
        //set module inputs
        var min = {};
        min[d.name] = value[0];
        var max = {};
        max[d.name] = value[1];
        module_input(min, update_success, error, module_id+"/min_value");
        module_input(max, update_success, error, module_id+"/max_value");
    }

    function range_query_create_slider(d, i) { 
        // var slider = d3.slider()
        //           .value([d.out_min,d.out_max])
        //           .axis(true)
        //           .min(d.in_min)
        //           .max(d.in_max)
        //           .orientation("horizontal")
        //           .on('slideend', function(evt, value) { range_query_input(evt, value, d); });
        var slider = d3.select(this)
                .append("input")
                .attr("type", "range")
                .attr("value", ""+d.in_min+","+d.in_max)
                .on("change", function(e) {
                    range_query_input(e, [e.valueLow, e.valueHigh], d);
                });
        sliders[d.name] = slider; // store the slider to reuse it later
    }

    function range_query_update_slider(d, i) {
        var slider = sliders[d.name];
        slider.value = ""+d.out_min+","+d.out_max;
    }

    function range_query_remove_slider(d, i) {
        delete sliders[d.name];
    }

    function update(data){
        module_update(data);

        if(data.ranges === undefined) { return; }

        //render sliders
        var containers = d3
                .select("#sliders")
                .selectAll(".slider")
                .data(data.ranges, function(d) { return d.name; });
        var cEnter = containers.enter()
                .append("div")
                .attr("class", "slider panel panel-default");

        cEnter.append("div")
            .attr("class", "slider-name panel-body")
            .text(function(d) { return "Attribute: '"+d.name+"'"; });
        
        cEnter.append("div")
            .attr("class", "slider-container");

        //containers.call(d3.slider().value([10,40]).orientation("vertical"));
        //the line above is simpler but does not work -- maybe because of a bug in d3-sliders. All thumbs are mixed up.
        cEnter.selectAll(".slider-container")
            .each(range_query_create_slider);

        // Update function
        containers.selectAll(".slider-container")
            .each(range_query_update_slider);

        // Exit function
        containers.exit()
            .each(range_query_remove_slider)
            .remove();
    }

    function update_success(){}

    function error(err){
        console.error(err);
    }

    return {
        ready: ready
    };
}();
