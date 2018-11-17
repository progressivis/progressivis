var multiclass2d_ready = (function() { // encapsulate to avoid side effects
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
        .on("zoom", multiclass2d_zoomed);

var view, gX, gY, zoomable;
var dataURL=null;
const DEFAULT_SIGMA = 0;
const DEFAULT_FILTER = "default";
const MAX_PREV_IMAGES = 3;
var imageHistory = new History(MAX_PREV_IMAGES);

const EPSILON = 1e-6;
function float_equal(a, b) {
    return Math.abs(a-b) < EPSILON;
}

function multiclass2d_update(data) {
    var wt = $('#module').attr('class').indexOf("active") >= 0;
    module_update(data, with_table=wt);
    multiclass2d_update_vis(data);
}

function _db(b) { return b[1]-b[0]; }

function multiclass2d_dragstart(d, i) {
    d3.event.sourceEvent.stopPropagation();
    d3.select(this).classed("dragging", true);
}

function multiclass2d_dragmove(d, i) {
    d[0] = xAxis.scale().invert(d3.event.x);
    d[1] = yAxis.scale().invert(d3.event.y);
    d3.select(this)
        .attr("cx", d3.event.x)
        .attr("cy", d3.event.y);
}

function multiclass2d_dragend(d, i) {
    var msg = {};
    d3.select(this).classed("dragging", false);
    msg[i] = d;
    module_input(msg, ignore, progressivis_error, module_id+"/move_point");
}

var node_drag = d3.drag()
        .on("start", multiclass2d_dragstart)
        .on("drag", multiclass2d_dragmove)
        .on("end", multiclass2d_dragend);


function multiclass2d_update_vis(rawdata) {
    var data = rawdata['sample'],
        bounds = rawdata['bounds'],
        ix, iy, iw, ih;

    if (!data || !bounds) return;
    var index = data['index'];
    var dot_color = ['red', 'blue', 'green', 'cyan', 'orange']
    if (prevBounds == null) { // first display, not refresh
        prevBounds = bounds;
        x.domain([bounds['xmin'], bounds['xmax']]).nice();
        y.domain([bounds['ymin'], bounds['ymax']]).nice();

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
            //.attr("xlink:href", rawdata['image'])
            .attr("xlink:href", dataURL)        
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
            transform = compute_transform(x, y,
                                          xAxis.scale(), yAxis.scale());
            svg.__zoom = transform; // HACK
            multiclass2d_zoomed(transform);
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
            //.attr("xlink:href", rawdata['image']);
            .attr("xlink:href", dataURL);        

        svg.select(".heatmapCompare")
            .attr("x", ix)
            .attr("y", iy)
            .attr("width",  iw)
            .attr("height", ih);
    }

    var spec =  {
        "data": { "url": "local" },
        "compose": {
            "mix": "max"
        },
        "rescale": {
        "type": "cbrt"
        },
        "legend":{}
        }
    data_ = rawdata['chart'];
    function render(spec, data) {
        var config = new MDM.Config(spec);
        config.loadJson(data).then(function () {
            var interp = new MDM.Interpreter(config);
            //var start = +new Date();
            interp.interpret();
            interp.render(document.getElementById('heatmapContainer'))/*.then(function(){
                dataURL = $("#heatmapContainer canvas")[0].toDataURL()
            })*/
            //var end = +new Date(); 
            //console.log('took ', (end - start), 'ms')
            interp = null;
        });
    }
    window.render = render;
    //MDM.Util.set_cache('JSON/local', data_);
    render(spec, data_);
    dataURL = $("#heatmapContainer canvas")[0].toDataURL()
    $("#heatmapContainer").html("")
    imageHistory.enqueueUnique(dataURL);

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
        .style("fill", function(d) { return dot_color[d[2]]}) //function(d) { return color(d.species); });
         .call(node_drag)
        .append("title")
        .text(function(d, i) { return index[i]; });
    dots .attr("cx", function(d) { return x(d[0]); })
         .attr("cy", function(d) { return y(d[1]); });
    dots.exit().remove();
    dots.order();
}

function multiclass2d_zoomed(t) {
    if (t === undefined)
        t = d3.event.transform;
    transform = t;
    gX.call(xAxis.scale(transform.rescaleX(x)));
    gY.call(yAxis.scale(transform.rescaleY(y)));
    zoomable.attr("transform", transform);
    svg.selectAll(".dot").attr("r", 3.5/transform.k);
}


    function multiclass2d_refresh(json) {
    if(json && json.payload) {
        multiclass2d_update(json.payload);}
    else {
        module_set_hotline(true);
        module_get(multiclass2d_update, error);
    }
        
}

function delta(d) { return d[1] - d[0]; }

function compute_transform(x, y, x0, y0) {
    var K0 = delta(x.domain()) / delta(x0.domain()),
        K1 = delta(y.domain()) / delta(y0.domain()),
        K = Math.min(K0, K1),
        X = -x(x0.invert(0))*K,
        Y = -y(y0.invert(0))*K;
    return d3.zoomIdentity
            .translate(X, Y)
            .scale(K);
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

function ignore(data) {}

function multiclass2d_filter() {
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

function multiclass2d_ready() {
    svg = d3.select("#multiclass_scatterplot svg")
         .attr("width", width + margin.left + margin.right)
         .attr("height", height + margin.top + margin.bottom);

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

    $('#filter').click(function() { multiclass2d_filter(); });
    
    refresh = multiclass2d_refresh; // function to call to refresh
    module_ready();
}
return multiclass2d_ready;
})();
