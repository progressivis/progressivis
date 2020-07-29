import {Config, Interpreter} from 'multiclass-density-maps';
import * as d3 from 'd3';
import History from "./history";
import * as colormaps from "./colormaps";
import {elementReady} from './es6-element-ready';
import $ from 'jquery';
//import * as ndarray_unpack from "ndarray-unpack";
var ndarray_unpack = require("ndarray-unpack");
var progressivis_data = null;
var ipyView = null;
var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom,
    svg, prevBounds = null, transform = d3.zoomIdentity;
var centroid_selection = {};
var collection_in_progress = false;
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
    //msg[i] = d;
    //module_input(msg, ignore, progressivis_error, module_id+"/move_point");
    //ipyView.model.set('move_point', msg);
    //ipyView.model.save_changes();
    //ipyView.touch();
    if(collection_in_progress){
	d3.select(this).style("fill", function(d) { return "green";});
	centroid_selection[i] = d;
    } else {
	msg[i] = d;
	ipyView.model.set('move_point', msg);
	ipyView.touch();
    }
}

var node_drag = d3.drag()
        .on("start", multiclass2d_dragstart)
        .on("drag", multiclass2d_dragmove)
        .on("end", multiclass2d_dragend);


function multiclass2d_update_vis(rawdata) {
    progressivis_data = rawdata;
    var bounds = rawdata['bounds'],
        ix, iy, iw, ih;
    if (!bounds) return;
    var st =  ipyView.model.get('samples');
    var index = [...Array(st.shape[0]*st.shape[2]).keys()];
    var rows = Array();
    for(let z in [...Array(st.shape[2])]){
	z = parseInt(z);
	for(let i in [...Array(st.shape[0])]){
	    rows.push([st.get(i,0,z), st.get(i,1,z),z]);
	}
    }
    var dot_color = ['red', 'blue', 'green', 'cyan', 'orange'];
    var data_ = rawdata['chart'];
    var hist_tensor = ipyView.model.get('hists');
    for(let s in data_['buffers']){
	let i = parseInt(s);
	var h_pick = hist_tensor.pick(null, null, i);
	data_['buffers'][i]['binnedPixels'] = ndarray_unpack(h_pick);
    }
    if(!window.spec){
        var spec =  {
            "data": { "url": "bar" },
            "compose": {
                "mix": "max"
            },
            "rescale": {
                "type": "cbrt"
            },
            "legend":{}
        };
        window.spec = spec;
    }
    function render(spec, data) {
        var config = new Config(spec);
        config.loadJson(data).then(function () {
            var interp = new Interpreter(config);
	    elementReady("#heatmapContainer").then((_)=>{
		interp.interpret();
		return interp.render(document.getElementById('heatmapContainer'));
	    });
        });
    }
    window.render = render;
    render(window.spec, data_);
    elementReady("#heatmapContainer canvas").then((that)=>{
    dataURL = $(that)[0].toDataURL();
    window.spec.data = {};
    imageHistory.enqueueUnique(dataURL);
    $('#map-legend').empty();
    $("#heatmapContainer svg").last().detach().appendTo('#map-legend');
    $("#heatmapContainer canvas").last().detach().attr('style','position: relative; left: -120px; top: 10px;').appendTo('#map-legend'); //blend
    $("#heatmapContainer").html("");
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
        //firstTime = false;
    }
    else { // prevBounds != null
        var changed = false;
	var v = null;
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
             .data(rows, function(d, i) { return index[i]; });
    
    dots.enter().append("circle")
         .attr("class", "dot")
         .attr("r", 3.5/transform.k)
         .attr("cx", function(d) { return x(d[0]); }) // use untransformed x0/y0
         .attr("cy", function(d) { return y(d[1]); })
         .style("fill", function(d) { return dot_color[d[2]]; })
         .call(node_drag)
        .append("title")
        .text(function(d, i) { return index[i]; });
    dots .attr("cx", function(d) { return x(d[0]); })
         .attr("cy", function(d) { return y(d[1]); }).style("fill", function(d) { return dot_color[d[2]]});
    dots.exit().remove();
	dots.order();
    });//end elementReady
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

/*
    function multiclass2d_refresh(json) {
    if(json && json.payload) {
        multiclass2d_update(json.payload);}
    else {
        module_set_hotline(true);
        module_get(multiclass2d_update, error);
    }
        
    }
*/
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

function multiclass2d_filter_debug() {
    ipyView.model.set('value', 333);
    //ipyView.model.save_changes();
    ipyView.touch();
}
function multiclass2d_filter() {
    console.log("call multiclass2d_filter");
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
    console.log("min:", min);
    console.log("max:", max);
    ipyView.model.set('value', {min: min, max: max});
    //ipyView.model.save_changes();
    ipyView.touch();
}

function move_centroids(){
    var txt = $('#init_centroids').html();
    if(txt == "Selection"){
        $('#init_centroids').html("Click when ready");
        collection_in_progress = true;
        centroid_selection = {};
	ipyView.model.set('modal', true);
    } else {
        $('#init_centroids').html("Selection");
        collection_in_progress = false;
        console.log(centroid_selection);
	ipyView.model.set('move_point', centroid_selection);
        centroid_selection = {};
	ipyView.model.set('modal', false);
    }
    ipyView.touch()	    
}

function multiclass2d_ready(view_) {
    ipyView = view_;
    svg = d3.select("#multiclass_scatterplot svg")
         .attr("width", width + margin.left + margin.right)
         .attr("height", height + margin.top + margin.bottom);

    $('#nav-tabs a').click(function (e) {
        e.preventDefault();
        $(this).tab('show');
    });
    prevBounds = null;
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
    colormaps.makeTableFilter(colorMap, "Default");
    $('#filter').unbind('click').click(function() { multiclass2d_filter(); });
    $('#init_centroids').click(function(d) { move_centroids(); });
}

export  {multiclass2d_update_vis as update_vis,  multiclass2d_ready as ready_, ipyView as view};
