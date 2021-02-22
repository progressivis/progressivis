'use strict';
import * as widgets from '@jupyter-widgets/base';
import _ from 'lodash';
import { elementReady } from './es6-element-ready';
const slpb = require('./sparkline_progressbar');
require('sorttable');
require('../css/sparkline-progressbar.css');
const jh = require('./layout_dict');

import { new_id } from './base';


const JsonHTMLModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: 'JsonHTMLModel',
    _view_name: 'JsonHTMLView',
    _model_module: 'progressivis-nb-widgets',
    _view_module: 'progressivis-nb-widgets',
    _model_module_version: '0.1.0',
    _view_module_version: '0.1.0',
    dom_id: 'json_html_dom_id',
    data: '{}',
    config: '{}',
  }),
});

// Custom View. Renders the widget model.
const JsonHTMLView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    let dom_id = this.model.get('dom_id');
    this.el.innerHTML = "<div id='" + dom_id + "'></div>";
    this.data_changed();
    // Observe changes in the value traitlet in Python, and define
    // a custom callback.
    this.model.on('change:config', this.data_changed, this);
    this.model.on('change:data', this.data_changed, this);
  },

  data_changed: function () {
    const that = this;
    const dom_id = this.model.get('dom_id');

    elementReady('#' + dom_id).then(() => jh.layout_dict_entry(that));
  },
});

const SparkLineProgressBarModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: 'SparkLineProgressBarModel',
    _view_name: 'SparkLineProgressBarView',
    _model_module: 'progressivis-nb-widgets',
    _view_module: 'progressivis-nb-widgets',
    _model_module_version: '0.1.0',
    _view_module_version: '0.1.0',
    data: '{}',
  }),
});

// Custom View. Renders the widget model.
const SparkLineProgressBarView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.id = 'sparklink-bp_' + new_id();
    this.el.innerHTML = `<div style="width: 100%;"><div class="slpb-bg">
<span id='${this.id}' class="slpb-fill" style="width: 70%;"></span>
</div></div>`;
    this.data_changed();
    // Observe changes in the value traitlet in Python, and define
    // a custom callback.
    this.model.on('change:data', this.data_changed, this);
  },
  data_changed: function () {
    const that = this;
    elementReady('#' + that.id).then(() => slpb.update_slpb(that));
  },
});

const PlottingProgressBarModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: 'PlottingProgressBarModel',
    _view_name: 'PlottingProgressBarView',
    _model_module: 'progressivis-nb-widgets',
    _view_module: 'progressivis-nb-widgets',
    _model_module_version: '0.1.0',
    _view_module_version: '0.1.0',
    data: '{}',
  }),
});

// Custom View. Renders the widget model.
const PlottingProgressBarView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.id = 'plotting-pb' + new_id();
    this.el.innerHTML = `<div style="width: 100%;">
         <div class="slpb-bg">
           <div id='${this.id}' class="slpb-fill" style="width: 70%;"></div>
         </div>
        </div>`;
    this.data_changed();
    // Observe changes in the value traitlet in Python, and define
    // a custom callback.
    this.model.on('change:data', this.data_changed, this);
  },

  data_changed: function () {
    let that = this;
    elementReady('#' + that.id).then(() => slpb.update_slpb(that));
  },
});

module.exports = {
  JsonHTMLModel: JsonHTMLModel,
  JsonHTMLView: JsonHTMLView,
  SparkLineProgressBarModel: SparkLineProgressBarModel,
  SparkLineProgressBarView: SparkLineProgressBarView,
  PlottingProgressBarModel: PlottingProgressBarModel,
  PlottingProgressBarView: PlottingProgressBarView,
};
