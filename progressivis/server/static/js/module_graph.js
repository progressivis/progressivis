"use strict";

var scheduler_run_number = -1;

const width=960, height=500;
const margin=10, pad=12;

function scheduler_get(success, error) {
    $.ajax({
	url: $SCRIPT_ROOT+'/progressivis/scheduler/?short=False',
	dataType: 'json',
	method: 'POST'
    })
	.done(success)
	.fail(error);
};

function graph_setup(){
  var outer = d3.select("#module-graph")
    .attr({width: width, height: height, "pointer-events": "all"});

  outer.append('rect')
    .attr({ class: 'background', width: "100%", height: "100%"})
    .call(d3.behavior.zoom().on("zoom", zoom));

  outer.append('svg:defs').append('svg:marker')
    .attr({
        id: 'end-arrow',
        viewBox: '0 -5 10 10',
        refX: 8,
        markerWidth: 6,
        markerHeight: 6,
        orient: 'auto'
    })
    .append('svg:path')
      .attr({
        d: 'M0,-5L10,0L0,5L2,0',
        'stroke-width': '0px',
        fill: '#000'});

   var vis = outer
   .append('g');

   function zoom(){
     vis.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")");
   }
}

/**
 * Similar to lodash's uniq, but allows comparing (nested) objects.
 */
function deep_uniq(coll){
  return _.reduce(coll, function(results, item) {
    return _.any(results, function(result) {
      return _.isEqual(result, item);
      }) ? results : results.concat([ item ]);
  }, []);
}

function scheduler_update(data) {
    scheduler_run_number = data['run_number'];
    graph_update_vis(data.modules);
}

/**
 * @param modules - list of modules in long format (including slot information)
 * @param name2id - map of progressivis module id to numerical index (cola requires numerical IDs)
 */
function collect_edges(modules, name2id){
  var retval = [];
  $.each(modules, function(i, module){
    $.each(module.output_slots, function(name, slot){
      if(slot){
        $.each(slot, function(j, link){
          retval.push({source: name2id[link.output_module], target: name2id[link.input_module]});
        });
      }
    });
  });
  
  //We draw a graph at the module (not slot) level, hence the need to remove possible duplicates.
  return deep_uniq(retval);
}

function graph_update_vis(modules){
  console.log("graph_update_vis");
  var nodes = _.map(modules, function(module, index){
    return { 
      id: index, 
      name: module.id,
      state: module.state
    };
  });

  var name2id = _.reduce(nodes, function(acc, node){
    acc[node.name] = node.id;
    return acc;
  }, {});

  var edges = collect_edges(modules, name2id);

  var d3cola = cola.d3adaptor().convergenceThreshold(0.1);

  d3cola
    .avoidOverlaps(true)
    .convergenceThreshold(1e-3)
    .flowLayout('x', 150)
    .size([width, height])
    .nodes(nodes)
    .links(edges)
    .jaccardLinkLengths(150);

  var vis = d3.select("#module-graph g");

  var node = vis.selectAll(".node")
    .data(nodes, function(d){ return d.id; })
    .enter().append("rect")
    .attr("class", function(d){ return "node " + d.state; })
    .attr({ rx: 5, ry: 5 });

    var label = vis.selectAll(".label")
    .data(nodes)
    .enter().append("text")
    .attr("class", "label")
    .text(function (d) { return d.name; })
    .each(function (d) {
            var b = this.getBBox();
            var extra = 2 * margin + 2 * pad;
            d.width = b.width + extra;
            d.height = b.height + extra;
    });

    var link = vis.selectAll(".link")
      .data(edges)
      .enter().append("path")
      .attr("class", "link");
  
    var lineFunction = d3.svg.line()
      .x(function (d) { return d.x; })
      .y(function (d) { return d.y; })
      .interpolate("linear");

    var routeEdges = function(){
      d3cola.prepareEdgeRouting();
      link.attr("d", function(d){ return lineFunction(d3cola.routeEdge(d)); });
    }

    d3cola.start(50, 100, 200).on("tick", function () {
      node.each(function (d) { d.innerBounds = d.bounds.inflate(-margin); })
        .attr("x", function (d) { return d.innerBounds.x; })
        .attr("y", function (d) { return d.innerBounds.y; })
        .attr("width", function (d) {
            return d.innerBounds.width();
        })
        .attr("height", function (d) { return d.innerBounds.height(); });
  
      label.attr("x", function(d){ return d.x; })
        .attr("y", function(d){ return d.y + (margin + pad) / 2; });

    }).on("end", routeEdges);

}

function graph_error(ev) {
  var contents = '<div class="alert alert-danger" role="alert">Server Error</div>';
  $('#error').html(contents);
}

function graph_refresh() {
  scheduler_get(scheduler_update, graph_error);
}

function scheduler_socketmsg(message) {
  var txt = message.data, run_number;
  if (txt.startsWith("tick ")) {
	run_number = Number(txt.substr(5));
	if (run_number > scheduler_run_number)
	  graph_refresh();
  } else { 
	  console.log('Scheduler received unexpected socket message: '+txt);
  }
}

function graph_ready() {
  graph_setup();
  graph_refresh();
  websocket_open("scheduler", scheduler_socketmsg);
}
