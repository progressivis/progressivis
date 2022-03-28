//'use strict';
import * as widgets from '@jupyter-widgets/base';
import _ from 'lodash';
import $ from 'jquery';
import { new_id } from './base';
import { elementReady } from './es6-element-ready';
//import * as colormaps from './colormaps';
import * as d3 from 'd3';
import History from './history';
import '../css/scatterplot.css';

const DEFAULT_SIGMA = 0;
const DEFAULT_FILTER = 'default';
const MAX_PREV_IMAGES = 3;

export const PrevImagesModel = widgets.DOMWidgetModel.extend(
  {
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
      _model_name: 'PrevImagesModel',
      _view_name: 'PrevImagesView',
      _model_module: 'progressivis-nb-widgets',
      _view_module: 'progressivis-nb-widgets',
      _model_module_version: '0.1.0',
      _view_module_version: '0.1.0',
      hists: ndarray([]),
      samples: ndarray([]),
      canvas_query: '',
    }),
  }
);

// Custom View. Renders the widget model.
export const PrevImagesView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.id = 'view_' + new_id();
    const previmgs = PrevImages(this);
    this.previmgs = previmgs;
    this.previmgs.template(this.el);
    this.model.on('msg:custom', this.data_changed, this);
  },
  data_changed: function () {
    console.log("data_changed");
    const qry = this.model.get('canvas_query');
    console.log(qry);
    this.previmgs.update_vis(qry);
  },
});


function PrevImages(ipyView) {
  const id = ipyView.id;
  let dataURL = null;
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
        <div class="form-inline">
          <div  id="historyGrp" style="height:80px;">
            <label>History</label>
            <table border="1"style="width:500px;height:80px;"><tr><td id="prevImages"></td></tr></table>
          </div>
          <br/>
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
  function _update_vis(qry) {
    elementReady(qry).then((that) => {
      dataURL = $(that)[0].toDataURL();
	imageHistory.enqueueUnique(dataURL);

      const prevImgElements = d3
        .select(swith_id('prevImages'))
        .selectAll('img')
        .data(imageHistory.getItems(), (d) => d);

      prevImgElements
        .enter()
        .append('img')
        .attr('width', 50)
        .attr('height', 50)


      prevImgElements
        .transition()
        .duration(50000)
        .attr('src', (d) => d)
        .attr('width', 100)
        .attr('height', 100);

      prevImgElements
        .exit()
        .transition()
        .duration(500)
        .attr('width', 5)
        .attr('height', 5)
        .remove();
    }); //end elementReady
  }
  function multiclass2d_ready() {
    svg = d3
      .select(swith_id('Scatterplot') + ' svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);

  return {
    update_vis: _update_vis,
    template: template,
    with_id: with_id
  };
}
