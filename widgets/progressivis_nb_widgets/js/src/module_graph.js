"use strict";
import * as d3 from 'd3';
import * as cola from 'webcola';
import $ from 'jquery';
//var module_graph = function(){
  var d3cola;
  var firstTime = true;
  
  const width=960, height=500;
  const margin=10, pad=12;

  
  function graph_setup() {
      var outer = d3.select("#module-graph")
              .attr('width', width)
              .attr('height', height)
              .attr("pointer-events", "all");
  
      outer.append('rect')
          .attr('class', 'background')
          .attr('width', "100%")
          .attr('height', "100%")
          .call(d3.zoom().on("zoom", zoom));
  
      var vis = outer.append('g');

      outer.append('svg:defs').append('svg:marker')
          .attr('id','end-arrow')
          .attr('viewBox','0 -5 10 10')
          .attr('refX',8)
          .attr('markerWidth',6)
          .attr('markerHeight',6)
          .attr('orient','auto')
          .append('svg:path')
          .attr('d','M0,-5L10,0L0,5L2,0Z')
          .attr('stroke-width','0px')
          .attr('fill','#000');
  
  
      function zoom() {
          vis.attr("transform", d3.event.transform);
      }
      d3cola = cola.d3adaptor(d3)
          .size([width, height])
          .avoidOverlaps(true)
          .convergenceThreshold(0.001)
          .flowLayout('x', 150)
          .jaccardLinkLengths(150);
  }
  
  /**
   * Similar to lodash's uniq, but allows comparing (nested) objects.
   */
  function deep_uniq(coll){
    return _.reduce(coll, function(results, item) {
      return _.some(results, function(result) {
        return _.isEqual(result, item);
        }) ? results : results.concat([ item ]);
    }, []);
  }
  
  function graph_update(data) {
      graph_update_vis(data.modules, firstTime);
      firstTime = false;
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
  
function graph_update_vis(modules, firstTime) {
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
  
      var vis = d3.select("#module-graph g");
  
      var node = vis.selectAll(".node")
              .data(nodes, function (d) { return d.id; });
  
  
      if (firstTime) {
          var edges = collect_edges(modules, name2id);

          node.enter().append("rect")
              .attr("class", function(d){ return "node " + d.state; })
              .attr('rx', 5)
              .attr('ry', 5);
  
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
                  .data(edges);

          var lineFunction = d3.line()
                  .x(function (d) { return d.x; })
                  .y(function (d) { return d.y; });
  
          var routeEdges = function(){
              d3cola.prepareEdgeRouting(margin / 3);

              link.enter().append("path")
                  .attr("class", "link")
                  .attr("d", function(d){ return lineFunction(d3cola.routeEdge(d)); });
          };
  
          d3cola.nodes(nodes)
              .links(edges)
              .start(50, 100, 200)
              .on("tick", function () {
                  var node = vis.selectAll(".node")
                          .data(nodes, function (d) { return d.id; })
                          .each(function (d) {
                      d.innerBounds = d.bounds.inflate(-margin);
                  })
                      .attr("x", function (d) {
                          return d.innerBounds.x;
                      })
                      .attr("y", function (d) {
                          return d.innerBounds.y;
                      })
                      .attr("width", function (d) {
                          return d.innerBounds.width();
                      })
                      .attr("height", function (d) {
                          return d.innerBounds.height();
                      });

                  link.attr("d", function(d) {
                      var route = cola.makeEdgeBetween(d.source.innerBounds, d.target.innerBounds, 5);
                      return lineFunction([route.sourceIntersection, route.arrowStart]);
                  });
                  label
                      .attr("x", function(d){ return d.x; })
                      .attr("y", function(d){ return d.y + (margin + pad) / 2; });
              })
              .on("end", routeEdges);
      }
      else {
          node.attr("class", function(d){ return "node " + d.state; });
      }
  }
  
/*
  return {
    ready: graph_ready,
    setup: graph_setup,
    update: graph_update,
    update_vis: graph_update_vis
  };
}();

module.exports =  {
    graph_update: graph_update,
    graph_setup: graph_setup
  };
*/
export {graph_update, graph_setup}
