var margin = {top: 20, right: 20, bottom: 30, left: 40},
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

var zoom = d3.behavior.zoom()
    .x(x)
    .y(y)
    .scaleExtent([1, 32])
    .on("zoom", scatterplot_zoomed);

const DEFAULT_SIGMA = 2;
const DEFAULT_FILTER = "default";


function scatterplot_update(data) {
    module_update(data);
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
        svg.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", width)
            .attr("height", height)
            .attr("fill", "black");

        var zoomable = svg.append("g").attr('id', 'zoomable');

        zoomable.append("image")
            .attr("class", "heatmap")
            .attr("xlink:href", rawdata['image'])
            .attr("preserveAspectRatio", "none")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", width)
            .attr("height", height)
            .attr("filter", "url(#gaussianBlur)");

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
            //.transition()
            //.duration(1000)
            .call(xAxis);

        svg.select(".y.axis")
            //.transition()
            //.duration(100)
            .call(yAxis);
    }

    var dots = d3.select("#zoomable")
            .selectAll(".dot")
             .data(data['data'], function(d, i) { return index[i]; });
    
    dots.enter().append("circle")
         .attr("class", "dot")
         .attr("r", 3.5)
         .attr("cx", function(d) { return x(d[0]); })
         .attr("cy", function(d) { return y(d[1]); })
         .style("fill", "blue") //function(d) { return color(d.species); });
        .append("title")
        .text(function(d, i) { return index[i]; });
    dots//.transition()  // Transition from old to new
        //.duration(500)  // Length of animation
         .attr("cx", function(d) { return x(d[0]); })
         .attr("cy", function(d) { return y(d[1]); });    
    dots.exit().remove();
}

function scatterplot_zoomed() {
    svg.select(".x.axis").call(xAxis);
    svg.select(".y.axis").call(yAxis);
    svg.select("#zoomable")
        .attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    svg.selectAll(".dot").attr("r", 3.5/d3.event.scale);
}


function scatterplot_refresh() {
  module_get(scatterplot_update, error);
}


/**
 * @param select - a select element that will be mutated 
 * @param names - list of option names (the value of an option is set to its name)
 */
function makeOptions(select, names){
  if(!select){
    console.warn("makeOptions requires an existing select element");
    return;
  }
  names.forEach(function(name){
    var option = document.createElement("option");
    option.setAttribute("value", name);
    option.innerHTML = name;
    select.appendChild(option);
  });
}

function scatterplot_ready() {
    svg = d3.select("#scatterplot svg")
         .attr("width", width + margin.left + margin.right)
         .attr("height", height + margin.top + margin.bottom)
        .append("g")
         .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
         .call(zoom);

    $('#nav-tabs a').click(function (e) {
        e.preventDefault();
        $(this).tab('show');
    });

    const gaussianBlur = document.getElementById("gaussianBlurElement");
    const filterSlider = $("#filterSlider");
    filterSlider.change(function(){
      gaussianBlur.setStdDeviation(this.value, this.value);
    });
    filterSlider.get(0).value = DEFAULT_SIGMA;
    gaussianBlur.setStdDeviation(DEFAULT_SIGMA, DEFAULT_SIGMA);

    const colorMap = document.getElementById("colorMap");
    const colorMapSelect = $("#colorMapSelect");
    colorMapSelect.change(function(){
      colormaps.makeTableFilter(colorMap, this.value);
    });
    colorMapSelect.get(0).value = DEFAULT_FILTER;
    makeOptions(colorMapSelect.get(0), colormaps.getTableNames());
    
    refresh = scatterplot_refresh; // function to call to refresh
    module_ready();
}
