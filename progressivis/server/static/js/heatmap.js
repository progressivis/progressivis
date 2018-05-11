var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom,
    svg, firstTime = true;

var x = d3.scaleLinear()
    .range([0, width]);

var y = d3.scaleLinear()
    .range([height, 0]);

var color = d3.scaleOrdinal(d3.schemeCategory10);

var xAxis = d3.axisBottom()
    .scale(x);

var yAxis = d3.axisLeft()
    .scale(y);


function heatmap_update(data) {
    module_update(data);
    heatmap_update_vis(data);
}

function heatmap_update_vis(data) {
    var image = data['image'],
        bounds = data['bounds'];

    if (!image || !bounds) return;
    x.domain([bounds['xmin'], bounds['xmax']]);
    y.domain([bounds['ymin'], bounds['ymax']]);

    if (firstTime) {
        svg.append("image")
            .attr("class", "heatmap")
            .attr("xlink:href", function() { return image; }) //+"&ts="+new Date().getTime(); })
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
            .text(data['columns'][0]);

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
            .attr("xlink:href", function() { return image+"&ts="+new Date().getTime(); });

        svg.select(".x.axis")
            .transition()
            .duration(1000)
            .call(xAxis);

        svg.select(".y.axis")
            .transition()
            .duration(100)
            .call(yAxis);
    }
}

function heatmap_refresh() {
  module_get(heatmap_update, error);
}

function heatmap_ready() {
    svg = d3.select("#heatmap")
        .append("svg")
         .attr("width", width + margin.left + margin.right)
         .attr("height", height + margin.top + margin.bottom)
        .append("g")
         .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    $('#nav-tabs a').click(function (e) {
        e.preventDefault();
        $(this).tab('show');
    });
    refresh = heatmap_refresh;
    module_ready();
}
