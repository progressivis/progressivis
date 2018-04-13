var scatterplot_ready = (function() {
var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom,
    svg, prevBounds = null, transform = d3.zoomIdentity;

var x     = d3.scaleLinear().range([0, width]),
    y     = d3.scaleLinear().range([height, 0]),
    color = d3.scaleOrdinal(d3.schemeCategory10),
    xAxis = d3.axisBottom(x)
        .tickSize(height)
        .tickPadding(8 - height),
    yAxis = d3.axisRight(y)
        .tickSize(width)
        .tickPadding(8 - width),
    zoom = d3.zoom()
        //.scaleExtent([1, 32])
        .on("zoom", scatterplot_zoomed);

var view, gX, gY, zoomable;

const DEFAULT_SIGMA = 1;
const DEFAULT_FILTER = "default";
const MAX_PREV_IMAGES = 3;
var imageHistory = new History(MAX_PREV_IMAGES);

function scatterplot_update(data) {
    module_update(data);
    scatterplot_update_vis(data);
}

function _db(b) { return b[1]-b[0]; }

function scatterplot_dragstart(d, i) {
    d3.event.sourceEvent.stopPropagation();
    d3.select(this).classed("dragging", true);
}

function scatterplot_dragmove(d, i) {
    d[0] = xAxis.scale().invert(d3.event.x);
    d[1] = yAxis.scale().invert(d3.event.y);
    d3.select(this)
        .attr("cx", d3.event.x)
        .attr("cy", d3.event.y);
}

function scatterplot_dragend(d, i) {
    var msg = {};
    d3.select(this).classed("dragging", false);
    msg[i] = d;
    module_input(msg, ignore, progressivis_error, module_id+"/move_point");
}

var node_drag = d3.drag()
        .on("start", scatterplot_dragstart)
        .on("drag", scatterplot_dragmove)
        .on("end", scatterplot_dragend);
/*
 We use a transform that is a bit tricky.
 All the coordinates are translated in pixels by the x0/y0 scales.
 The x/y scales are the zoomed versions managed by the zoom tool.
 However, the zoom changes the transform of the group that contains
 the heatmap and the points in screen coordinates.
 We change the radius of the points to be constant, so we divide them by the
 zoom scale.
 We position everything on the group according to x0/y0 and not x/y, which are
 only used by the axes.

 When a new scatterplot arrives, the bounds can have changed. When
 progressing, the bounds can grow, but through filtering, the bounds
 can also shrink.  So when a new scatterplot arrives, we recompute the
 scale/translation of the zoom component to show the same viewport as
 before.
*/
function scatterplot_update_vis(rawdata) {
    var data = rawdata['scatterplot'],
        bounds = rawdata['bounds'],
        ix, iy, iw, ih;

    if (!data || !bounds) return;
    var index = data['index'];

    var x0 = xAxis.scale(),
        y0 = yAxis.scale();


    if (prevBounds == null) { // first display, not refresh
        prevBounds = bounds;
        x.domain([bounds['xmin'], bounds['xmax']]).nice();
        y.domain([bounds['ymin'], bounds['ymax']]).nice();

        // svg.append("rect")
        //     .attr("x", 0)
        //     .attr("y", 0)
        //     .attr("width", width)
        //     .attr("height", height)
        //     .style("fill", "none")
        //     .style("pointer-events", "all");

        zoomable = svg.append("g")
            .attr('id', 'zoomable')
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        ix = x(bounds['xmin']),
        iy = y(bounds['ymax']),
        iw = x(bounds['xmax'])-ix,
        ih = y(bounds['ymin'])-iy;
            
        zoomable.append("image")
            .attr("class", "heatmap")
            .style("pointer-events",  "none")
            .attr("xlink:href", rawdata['image'])
            .attr("preserveAspectRatio", "none")
            .attr("x", ix)
            .attr("y", iy)
            .attr("width",  iw)
            .attr("height", ih)
            .attr("filter", "url(#gaussianBlur)");

        svg.append("image")
            .attr("class", "heatmapCompare")
            .style("pointer-events",  "none")
            .attr("preserveAspectRatio", "none")
            .attr("opacity", 0.5)
            .attr("x", ix)
            .attr("y", iy)
            .attr("width",  iw)
            .attr("height", ih);

        gX = svg.append("g")
            .attr("class", "x axis axis--x")
            .call(xAxis);

        gY = svg.append("g")
            .attr("class", "y axis axis--y")
            .call(yAxis);

        svg.call(zoom);
        firstTime = false;
    }
    else { // prevBounds != null
        var changed = false;
        for (v in prevBounds) {
            if (prevBounds[v] != bounds[v]) {
                changed = true;
                break;
            }
        }
        if (changed) {
            var x_bounds = [prevBounds.xmin, prevBounds.xmax],
                y_bounds = [prevBounds.ymin, prevBounds.ymax];

            console.log('Bounds have changed');
            prevBounds = bounds;
            x.domain([bounds['xmin'], bounds['xmax']]).nice();
            y.domain([bounds['ymin'], bounds['ymax']]).nice();
            transform = compute_transform(x, y, x0, y0);
            scatterplot_zoomed(transform);
        }

        ix = x(bounds['xmin']);
        iy = y(bounds['ymax']);
        iw = x(bounds['xmax'])-ix;
        ih = y(bounds['ymin'])-iy;
        svg.select(".heatmap")
            .attr("x", ix)
            .attr("y", iy)
            .attr("width",  iw)
            .attr("height", ih)
            .attr("xlink:href", rawdata['image']);

        svg.select(".heatmapCompare")
            .attr("x", ix)
            .attr("y", iy)
            .attr("width",  iw)
            .attr("height", ih);
        
        //zoom.event(svg); // propagate
    }
    var imgSrc = rawdata['image'];
    imageHistory.enqueueUnique(imgSrc);

    var prevImgElements = d3.select("#prevImages")
            .selectAll("img")
            .data(imageHistory.getItems(), function(d){ return d; });

    prevImgElements.enter()
        .append("img")
        .attr("width", 50)
        .attr("height", 50)
        .on("mouseover", function(d){
          d3.select(".heatmapCompare")
            .attr("xlink:href", d)
            .attr("visibility", "inherit");
        })
        .on("mouseout", function(d){
          d3.select(".heatmapCompare")
            .attr("visibility", "hidden");
        });

    prevImgElements.transition().duration(500)
        .attr("src", function(d){ return d; })
        .attr("width", 100)
        .attr("height", 100);

    prevImgElements.exit().transition().duration(500)
        .attr("width", 5)
        .attr("height", 5)
        .remove();

    var dots = zoomable
            .selectAll(".dot")
             .data(data['data'], function(d, i) { return index[i]; });
    
    dots.enter().append("circle")
         .attr("class", "dot")
         .attr("r", 3.5/transform.k)
         .attr("cx", function(d) { return x(d[0]); }) // use untransformed x0/y0
         .attr("cy", function(d) { return y(d[1]); })
         .style("fill", "blue") //function(d) { return color(d.species); });
         .call(node_drag)
        .append("title")
        .text(function(d, i) { return index[i]; });
    dots .attr("cx", function(d) { return x(d[0]); })
         .attr("cy", function(d) { return y(d[1]); });
    dots.exit().remove();
    dots.order();
}

function scatterplot_zoomed(t) {
    if (t === undefined)
        t = d3.event.transform;
    transform = t;
    gX.call(xAxis.scale(transform.rescaleX(x)));
    gY.call(yAxis.scale(transform.rescaleY(y)));
    zoomable.attr("transform", transform);
    svg.selectAll(".dot").attr("r", 3.5/transform.k);
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

function compute_transform(x, y, x0, y0) {
    var X = x(x0.invert(0)),
        Y = y(y0.invert(height)),
        K = (x(x0.invert(width))-X) / width;
    return new d3.zoomIdentity.translate(X, Y).scale(K);
}

function ignore(data) {}

function scatterplot_filter() {
    var xscale = xAxis.scale(),
        xmin = xscale.invert(0),
        xmax = xscale.invert(width),
        yscale = yAxis.scale(),
        ymin = yscale.invert(height),
        ymax = yscale.invert(0),
        bounds  = prevBounds,
        columns = progressivis_data['columns'],
        min     = {},
        max     = {};

    
    if (xmin != bounds['xmin'])
        min[columns[0]] = xmin;
    else
        min[columns[0]] = null; // NaN means bump to min
    if (xmax != bounds['xmax']) 
        max[columns[0]] = xmax;
    else
        max[columns[0]] = null;

    if (ymin != bounds['ymin'])
        min[columns[1]] = ymin;
    else
        min[columns[1]] = null;

    if (ymax != bounds['ymax'])
        max[columns[1]] = ymax;
    else
        max[columns[1]] = null;
    
    module_input(min, ignore, progressivis_error, module_id+"/min_value");
    module_input(max, ignore, progressivis_error, module_id+"/max_value");
}

function scatterplot_ready() {
    svg = d3.select("#scatterplot svg")
         .attr("width", width + margin.left + margin.right)
         .attr("height", height + margin.top + margin.bottom);
        //.append("g")
        //.attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        //.call(zoom);

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

    $('#filter').click(function() { scatterplot_filter(); });
    
    refresh = scatterplot_refresh; // function to call to refresh
    module_ready();
}
return scatterplot_ready;
})();
