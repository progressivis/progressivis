'use strict';
import * as d3 from 'd3';
import * as cola from 'webcola';
import $ from 'jquery';
import _ from 'lodash';

export function module_graph(view) {
  const id = view.id;
  let d3cola;
  let firstTime = true;

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
    d3cola = cola
      .d3adaptor(d3)
      .size([width, height])
      .avoidOverlaps(true)
      .convergenceThreshold(0.001)
      .flowLayout('x', 150)
      .jaccardLinkLengths(150);
  }

  /**
   * Similar to lodash's uniq, but allows comparing (nested) objects.
   */
  function deep_uniq(coll) {
    return _.reduce(
      coll,
      (results, item) => _.some(results, result => _.isEqual(result, item))
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
      state: module.state
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

    const node = vis.selectAll('.node').data(nodes, d => d.id);

    if (firstTime) {
      const edges = collect_edges(modules, name2id);

      node
        .enter()
        .append('rect')
        .attr('class', d => 'node ' + d.state)
        .attr('rx', 5)
        .attr('ry', 5);

      const label = vis
        .selectAll('.label')
        .data(nodes)
        .enter()
        .append('text')
        .attr('class', 'label')
        .text(d => d.name)
        .each(function (d) {
          const b = this.getBBox();
          const extra = 2 * margin + 2 * pad;
          d.width = b.width + extra;
          d.height = b.height + extra;
        });

      const link = vis.selectAll('.link').data(edges);

      const lineFunction = d3
        .line()
        .x(d => d.x)
        .y(d => d.y);

      const routeEdges = () => {
        d3cola.prepareEdgeRouting(margin / 3);

        link
          .enter()
          .append('path')
          .attr('class', 'link')
          .attr('d', d => lineFunction(d3cola.routeEdge(d)));
      };

      d3cola
        .nodes(nodes)
        .links(edges)
        .start(50, 100, 200)
        .on('tick', () => {
          vis
            .selectAll('.node')
            .data(nodes, d => d.id)
            .each(d => {
              d.innerBounds = d.bounds.inflate(-margin);
            })
            .attr('x', d => d.innerBounds.x)
            .attr('y', d => d.innerBounds.y)
            .attr('width', d => d.innerBounds.width())
            .attr('height', d => d.innerBounds.height());

          link.attr('d', d => {
            const route = cola.makeEdgeBetween(
              d.source.innerBounds,
              d.target.innerBounds,
              5
            );
            return lineFunction([route.sourceIntersection, route.arrowStart]);
          });
          label
            .attr('x', d => d.x)
            .attr('y', d => d.y + (margin + pad) / 2);
        })
        .on('end', routeEdges);
    } else {
      node.attr('class', d => 'node ' + d.state);
    }
  }

  return {
    ready: graph_ready,
    update_vis: graph_update,
  };
}
