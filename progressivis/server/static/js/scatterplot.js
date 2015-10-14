var scatterplot_status = null,
    margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom,
    svg, firstTime = true;

var x = d3.scale.linear()
    .range([0, width]);

var y = d3.scale.linear()
    .range([height, 0]);

var color = d3.scale.category10();

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left");


function scatterplot_update(data) {
    module_status = data;
    module_run_number = data.last_update;
    module_update_table(data);
    scatterplot_update_vis(data);
}

function scatterplot_update_vis(rawdata) {
    var data = rawdata['scatterplot'],
        bounds = rawdata['bounds'];

    if (!data || !bounds) return;
    var index = data['index'];
    x.domain([bounds['xmin'], bounds['xmax']]).nice();
    y.domain([bounds['ymin'], bounds['ymax']]).nice();

    if (firstTime) {
        svg.append("image")
            .attr("class", "heatmap")
            .attr("xlink:href", rawdata['image'])
            .attr("preserveAspectRatio", "none")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", width)
            .attr("height", height);

        svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis)
            .append("text")
            .attr("class", "label")
            .attr("x", width)
            .attr("y", -6)
            .style("text-anchor", "end")
            .text([0]);

        svg.append("g")
            .attr("class", "y axis")
            .call(yAxis)
            .append("text")
            .attr("class", "label")
            .attr("transform", "rotate(-90)")
            .attr("y", 6)
            .attr("dy", ".71em")
            .style("text-anchor", "end")
            .text(data['columns'][1]);
        firstTime = false;
    }
    else { // not firstTime
        svg.select(".heatmap")
            .attr("xlink:href", rawdata['image']);

        svg.select(".x.axis")
            .transition()
            .duration(1000)
            .call(xAxis);

        svg.select(".y.axis")
            .transition()
            .duration(100)
            .call(yAxis);
    }

    var dots = svg.selectAll(".dot")
            .data(data['data'], function(d, i) { return index[i]; });
    
    dots.enter().append("circle")
         .attr("class", "dot")
         .attr("r", 3.5)
         .attr("cx", function(d) { return x(d[0]); })
         .attr("cy", function(d) { return y(d[1]); })
         .style("fill", "blue") //function(d) { return color(d.species); });
        .append("title")
        .text(function(d, i) { return index[i]; });
    dots.transition()  // Transition from old to new
        .duration(500)  // Length of animation
         .attr("cx", function(d) { return x(d[0]); })
         .attr("cy", function(d) { return y(d[1]); });    
    dots.exit().remove();
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
    svg = d3.select("#scatterplot")
        .append("svg")
         .attr("width", width + margin.left + margin.right)
         .attr("height", height + margin.top + margin.bottom)
        .append("g")
         .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    scatterplot_refresh();
    $('#nav-tabs a').click(function (e) {
        e.preventDefault();
        $(this).tab('show');
    });
    websocket_open("module "+module_id, scatterplot_socketmsg);
}
