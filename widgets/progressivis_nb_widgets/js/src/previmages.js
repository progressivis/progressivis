//'use strict';
import * as widgets from '@jupyter-widgets/base';
import _ from 'lodash';
import $ from 'jquery';
import { new_id } from './base';
import { elementReady } from './es6-element-ready';
import * as colormaps from './colormaps';
import * as d3 from 'd3';
import History from './history';
import '../css/scatterplot.css';

const DEFAULT_SIGMA = 0;
const DEFAULT_FILTER = 'default';
const MAX_PREV_IMAGES = 3;

export const PrevImagesModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: 'PrevImagesModel',
    _view_name: 'PrevImagesView',
    _model_module: 'progressivis-nb-widgets',
    _view_module: 'progressivis-nb-widgets',
    _model_module_version: '0.1.0',
    _view_module_version: '0.1.0',
    hists: ndarray([]),
    samples: ndarray([]),
    target: '',
  }),
});

// Custom View. Renders the widget model.
export const PrevImagesView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.id = 'view_' + new_id();
    const previmgs = PrevImages(this);
    this.previmgs = previmgs;
    this.previmgs.template(this.el);
    let that = this;
    elementReady('#' + previmgs.with_id('prevImages')).then(() =>
      previmgs.ready(that.model.get('target'))
    );
    this.model.on('msg:custom', this.data_changed, this);
  },
  data_changed: function () {
    console.log('data_changed');
    const target = this.model.get('target');
    console.log(target);
    this.previmgs.update_vis(target);
  },
});

function PrevImages(ipyView) {
  const id = ipyView.id;
  let dataURL = null;
  let firstTime = true;
  let zoomable;
  let svg;
  let bounds = null;
  const imageHistory = new History(MAX_PREV_IMAGES);
  function with_id(prefix) {
    return prefix + '_' + id;
  }
  function s(id) {
    return '#' + id;
  }
  function swith_id(id) {
    return s(with_id(id));
  }

  function template(element) {
    let temp = document.querySelector('#PrevImages');
    if (temp === null) {
      // Install the template as a dom template node
      temp = document.createElement('template');
      temp.setAttribute('id', 'PrevImages');
      temp.innerHTML = `<div class="tab-content">
    <div >
      <div id=''>
        <svg>
          <filter id="gaussianBlur" width="100%" height="100%" x="0" y="0">
            <feGaussianBlur id="gaussianBlurElement" in="SourceGraphic" stdDeviation="0" />
            <feComponentTransfer id="colorMap">
              <feFuncR type="table" tableValues="1 1 1"/>
              <feFuncG type="table" tableValues="0.93 0.001 0"/>
              <feFuncB type="table" tableValues="0.63 0.001 0"/>
            </feComponentTransfer>
          </filter>
        </svg>
        <div class="form-inline">
          <div  id="historyGrp" style="height:80px;">
            <label>History</label>
            <table style="height:120px;border:1px solid black;border-collapse: collapse;"><tr><td width='300px' id="prevImages"></td>
<td>Blur radius</td><td><input class="form-control" id="filterSlider" type="range" value="0" min="0" max="5" step="0.1"></input></td>
<td>Color map</td><td><select id="colorMapSelect" class="form-control"></select></td>
</tr></table>
          </div>


          </div>
          <br/>
        </div>
       </div>
    </div>
  </div>

`;
      document.body.appendChild(temp);
    }
    const templateClone = temp.content.cloneNode(true);
    // Rename all the ids to be unique
    const with_ids = templateClone.querySelectorAll('[id]');
    const ids = new Set();

    for (const element of with_ids) {
      const eid = element.id ? with_id(element.id) : with_id('PrevImages');
      if (ids.has(eid)) {
        console.log(`Error in PrevImages.template(), duplicate id '${eid}'`);
        // TODO fix it
      }
      element.id = eid;
    }

    element.appendChild(templateClone);
  }
  //https://github.com/jupyter-widgets/ipywidgets/issues/1840
  function _update_vis(target) {
    let targetP = '.' + target;
    let targetCanvas = targetP + ' canvas';
    let targetSvg = targetP + ' svg';
    elementReady(targetCanvas).then((that) => {
      if (firstTime) {
        let w = $(targetCanvas).first().attr('width');
        let h = $(targetCanvas).first().attr('height');
        _createSvg(w, h);
        firstTime = false;
      }
      dataURL = $(that)[0].toDataURL();
      $(targetCanvas).hide();
      imageHistory.enqueueUnique(dataURL);
      let svgQry = swith_id('PrevImages') + ' svg';
      svg.select(svgQry + ' .heatmap').attr('xlink:href', dataURL);
      const prevImgElements = d3
        .select(swith_id('prevImages'))
        .selectAll('img')
        .data(imageHistory.getItems(), (d) => d);
      prevImgElements
        .enter()
        .append('img')
        .attr('width', 50)
        .attr('height', 50)
        .on('mouseover', (d) => {
          d3.select(`${svgQry} .heatmapCompare`)
            .attr('xlink:href', d)
            .attr('visibility', 'inherit');
        })
        .on('mouseout', () => {
          d3.select(`${svgQry} .heatmapCompare`).attr('visibility', 'hidden');
        });
      prevImgElements
        .transition()
        .duration(500)
        .attr('src', (d) => d)
        .attr('width', 100)
        .attr('height', 100);
      let lenImgs = $(swith_id('prevImages') + ' img').length;
      let exitDuration = 500 - 100 * Math.max(lenImgs - 3, 0);
      prevImgElements
        .exit()
        .transition()
        .duration(exitDuration)
        .attr('width', 5)
        .attr('height', 5)
        .remove();
    }); //end elementReady
  }

  /**
   * @param select - a select element that will be mutated
   * @param names - list of option names (the value of an option is set to its name)
   */
  function makeOptions(select, names) {
    if (!select) {
      console.warn('makeOptions requires an existing select element');
      return;
    }
    names.forEach((name) => {
      const option = document.createElement('option');
      option.setAttribute('value', name);
      option.innerHTML = name;
      select.appendChild(option);
    });
  }

  function _createSvg(w, h) {
    console.log('bounds', bounds);
    svg = d3
      .select(swith_id('PrevImages') + ' svg')
      .attr('width', w)
      .attr('height', h);
    zoomable = svg.append('g').attr('id', with_id('zoomable'));
    zoomable
      .append('image')
      .attr('class', 'heatmap')
      .style('pointer-events', 'none')
      .attr('xlink:href', dataURL)
      .attr('preserveAspectRatio', 'none')
      .attr('width', w)
      .attr('height', h)
      .attr('filter', `url(${swith_id('gaussianBlur')})`);
    svg
      .append('image')
      .attr('class', 'heatmapCompare')
      .style('pointer-events', 'none')
      .attr('preserveAspectRatio', 'none')
      .attr('opacity', 0.5)
      .attr('width', w)
      .attr('height', h);
    const gaussianBlur = document.getElementById(
      with_id('gaussianBlurElement')
    );
    const filterSlider = $(swith_id('filterSlider'));
    filterSlider.change(function () {
      const value = this.value;
      gaussianBlur.setStdDeviation(value, value);
    });
    filterSlider.get(0).value = DEFAULT_SIGMA;
    gaussianBlur.setStdDeviation(DEFAULT_SIGMA, DEFAULT_SIGMA);
    const colorMap = document.getElementById(with_id('colorMap'));
    const colorMapSelect = $(swith_id('colorMapSelect'));
    colorMapSelect.change(function () {
      colormaps.makeTableFilter(colorMap, this.value);
    });
    colorMapSelect.get(0).value = DEFAULT_FILTER;
    makeOptions(colorMapSelect.get(0), colormaps.getTableNames());
    colormaps.makeTableFilter(colorMap, 'Default');
  }
  function _ready(t) {}
  return {
    ready: _ready,
    update_vis: _update_vis,
    template: template,
    with_id: with_id,
  };
}
