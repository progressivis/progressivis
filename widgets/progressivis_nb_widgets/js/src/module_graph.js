'use strict';
import * as widgets from '@jupyter-widgets/base';
import * as d3 from 'd3';
import * as cola from 'webcola';
import $ from 'jquery';
import _ from 'lodash';
import { new_id } from './base';
import { elementReady } from './es6-element-ready';

import '../css/module-graph.css';

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
export const ModuleGraphModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: 'ModuleGraphModel',
    _view_name: 'ModuleGraphView',
    _model_module: 'progressivis-nb-widgets',
    _view_module: 'progressivis-nb-widgets',
    _model_module_version: '0.1.0',
    _view_module_version: '0.1.0',
    data: 'Hello ModuleGraph!',
  }),
});


// Custom View. Renders the widget model.
export const ModuleGraphView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.id = 'module_graph_' + new_id();
    this.module_graph = module_graph(this);
    this.el.innerHTML = `<svg id="${this.id}" width="960" height="500"></svg>`;
    const that = this;
    elementReady('#' + this.id).then(() => {
      that.module_graph.ready();
      that.data_changed();
    });
    console.log('Render ModuleGraphView');
    // Observe changes in the value traitlet in Python, and define
    // a custom callback.
    this.model.on('change:data', this.data_changed, this);
  },

  data_changed: function () {
    console.log('Data changed ModuleGraphView');
    let val = this.model.get('data');
    if (val == '{}') return;
    this.module_graph.update_vis(JSON.parse(val));
  },
});

function eqSet(as, bs) {
    if (as.size !== bs.size) return false;
    for (var a of as) if (!bs.has(a)) return false;
    return true;
}

function module_graph(view) {
  const id = view.id;
  let firstTime = true;
  let prev_edges = [];

  const width = 960,
    height = 500;
  const margin = 10,
    pad = 12;

  function graph_ready() {
    const outer = d3
      .select('#' + id)
      .attr('width', width)
      .attr('height', height)
      .attr('pointer-events', 'all');

    outer
      .append('rect')
      .attr('class', 'background')
      .attr('width', '100%')
      .attr('height', '100%')
      .call(d3.zoom().on('zoom', zoom));

    const vis = outer.append('g');

    outer
      .append('svg:defs')
      .append('svg:marker')
      .attr('id', 'end-arrow') // used from module_graph.css
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('svg:path')
      .attr('d', 'M0,-5L10,0L0,5L2,0Z')
      .attr('stroke-width', '0px')
      .attr('fill', '#000');

    function zoom() {
      vis.attr('transform', d3.event.transform);
    }
  }

  /**
   * Similar to lodash's uniq, but allows comparing (nested) objects.
   */
  function deep_uniq(coll) {
    return _.reduce(
      coll,
      (results, item) =>
        _.some(results, (result) => _.isEqual(result, item))
          ? results
          : results.concat([item]),
      []
    );
  }

  function graph_update(data) {
    graph_update_vis(data.modules, firstTime);
    firstTime = false;
  }

  /**
   * @param modules - list of modules in long format (including slot information)
   * @param name2id - map of progressivis module id to numerical index (cola requires numerical IDs)
   */
  function collect_edges(modules, name2id) {
    const retval = [];
    $.each(modules, (i, module) => {
      $.each(module.output_slots, (name, slot) => {
        if (slot) {
          $.each(slot, (j, link) => {
            retval.push({
              source: name2id[link.output_module],
              target: name2id[link.input_module],
              id: link.output_module+":"+link.input_module
            });
          });
        }
      });
    });

    //We draw a graph at the module (not slot) level, hence the need to remove possible duplicates.
    return deep_uniq(retval);
  }

  function graph_update_vis(modules, firstTime) {
    const nodes = _.map(modules, (module, index) => ({
      id: index,
      name: module.id,
      state: module.state,
    }));

    const name2id = _.reduce(
      nodes,
      (acc, node) => {
        acc[node.name] = node.id;
        return acc;
      },
      {}
    );

    const vis = d3.select('#' + id + ' g');

    const edges = collect_edges(modules, name2id)
          .sort(function (a, b) {
            if (a.source < b.source) return -1;
            if (a.source > b.source) return 1;
            if (a.target < b.target) return -1;
            if (a.target > b.target) return 1;
            return 0;
          });

    const node = vis.selectAll('.node').data(nodes, (d) => d.name);
    node
      .exit()
      .remove();
    node
      .enter()
      .append('rect')
      .attr('class', (d) => 'node ' + d.state)
      .attr('rx', 5)
      .attr('ry', 5);
    node
      .attr('class', (d) => 'node ' + d.state);

    const label = vis
          .selectAll('.label')
          .data(nodes, (d) => d.name);
    label
        .exit().remove();
    label
        .enter()
        .append('text')
        .attr('class', 'label')
        .text((d) => d.name)
        .each(function (d) {
          const b = this.getBBox();
          const extra = 2 * margin + 2 * pad;
          d.width = b.width + extra;
          d.height = b.height + extra;
        });
    label
        .each(function (d) {
          const b = this.getBBox();
          const extra = 2 * margin + 2 * pad;
          d.width = b.width + extra;
          d.height = b.height + extra;
        });


    if (! _.isEqual(edges, prev_edges)) {
      const d3cola = cola
            .d3adaptor(d3)
            .size([width, height])
            .avoidOverlaps(true)
            .convergenceThreshold(0.001)
            .flowLayout('x', 150)
            .jaccardLinkLengths(150);
      prev_edges = edges;
      console.log("Laying out graph");

      const link = vis.selectAll('.link').data(edges, (d) => d.id);
      link.exit().remove();
      link.enter()
        .append('path')
        .attr('class', 'link')
        .attr('d', "");

      const lineFunction = d3.line().x((d) => d.x).y((d) => d.y);
      const routeEdges = () => {
        vis.selectAll('.node')
          .data(nodes, (d) => d.name)
          .each((d) => { d.innerBounds = d.bounds.inflate(-margin); })
          .attr('x', (d) => d.innerBounds.x)
          .attr('y', (d) => d.innerBounds.y)
          .attr('width', (d) => d.innerBounds.width())
          .attr('height', (d) => d.innerBounds.height());

        vis.selectAll('.link').data(edges, (d) => d.id)
          .attr('d', (d) => {
            const route = cola.makeEdgeBetween(
              d.source.innerBounds,
              d.target.innerBounds,
              5
            );
            return lineFunction([route.sourceIntersection, route.arrowStart]);
          });
        vis.selectAll('.label')
          .data(nodes, (d) => d.name)
          .attr('x', (d) => d.x)
          .attr('y', (d) => d.y + (margin + pad) / 2);

        d3cola.prepareEdgeRouting(margin / 3);

        const link = vis.selectAll('.link')
              .data(edges, (d) => d.id)
              .attr('d', (d) => lineFunction(d3cola.routeEdge(d)));
      };

      try {
        d3cola
          .nodes(nodes)
          .links(edges)
          .start(50, 100, 200)
          .on('end', routeEdges);
      }
      catch(e) {
        console.log("Error in routEdges: "+e);
      }
    }
  }

  return {
    ready: graph_ready,
    update_vis: graph_update,
  };
}
