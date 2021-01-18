'use strict';
import * as widgets from '@jupyter-widgets/base';
import _ from 'lodash';
import { elementReady } from './es6-element-ready';
const dt = require('./data_table');
const slpb = require('./sparkline_progressbar');
require('sorttable');
require('../css/module-graph.css');
require('../css/scatterplot.css');
require('../css/sparkline-progressbar.css');
require('datatables/media/css/jquery.dataTables.css');
import { SensitiveHTML } from './sensitive_html';
const jh = require('./layout_dict');

import { Scatterplot } from './scatterplot';
import { module_graph } from './module_graph';

import {
  data_union_serialization,
  listenToUnion,
} from 'jupyter-dataserializers';

const ndarray = require('ndarray');
window.ndarray = ndarray;
// See example.py for the kernel counterpart to this file.

let newid = 1;

function new_id() {
  return newid++;
}

// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including
//
//  - `_view_name`
//  - `_view_module`
//  - `_view_module_version`
//
//  - `_model_name`
//  - `_model_module`
//  - `_model_module_version`
//
//  when different from the base class.

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
const ScatterplotModel = widgets.DOMWidgetModel.extend(
  {
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
      _model_name: 'ScatterplotModel',
      _view_name: 'ScatterplotView',
      _model_module: 'progressivis-nb-widgets',
      _view_module: 'progressivis-nb-widgets',
      _model_module_version: '0.1.0',
      _view_module_version: '0.1.0',
      hists: ndarray([]),
      samples: ndarray([]),
      data: 'Hello Scatterplot!',
      value: '{0}',
      move_point: '{0}',
      modal: false,
      to_hide: [],
    }),
  },
  {
    serializers: _.extend(
      {
        hists: data_union_serialization,
        samples: data_union_serialization,
      },
      widgets.DOMWidgetModel.serializers
    ),
  }
);

// Custom View. Renders the widget model.
const ScatterplotView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.id = 'view_' + new_id();
    const scatterplot = Scatterplot(this);
    this.scatterplot = scatterplot;
    this.scatterplot.template(this.el);
    let that = this;
    elementReady('#' + scatterplot.with_id('prevImages')).then(() =>
      scatterplot.ready(that)
    );
    listenToUnion(this.model, 'hists', this.update.bind(this), true);
    listenToUnion(this.model, 'samples', this.update.bind(this), true);
    // Observe changes in the value traitlet in Python, and define
    // a custom callback.
    this.model.on('change:data', this.data_changed, this);
  },
  data_changed: function () {
    //console.log("data_changed");
    const val = this.model.get('data');
    this.scatterplot.update_vis(JSON.parse(val));
  },
});

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
const ModuleGraphModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: 'ModuleGraphModel',
    _view_name: 'ModuleGraphView',
    _model_module: 'progressivis-nb-widgets',
    _view_module: 'progressivis-nb-widgets',
    _model_module_version: '0.1.0',
    _view_module_version: '0.1.0',
    data: 'Hello ModuleGraph!',
    //value: '{0}'
  }),
});

// Custom View. Renders the widget model.
const ModuleGraphView = widgets.DOMWidgetView.extend({
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
    console.log('REnder ModuleGraphView');
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

const SensitiveHTMLModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: 'SensitiveHTMLModel',
    _view_name: 'SensitiveHTMLView',
    _model_module: 'progressivis-nb-widgets',
    _view_module: 'progressivis-nb-widgets',
    _model_module_version: '0.1.0',
    _view_module_version: '0.1.0',
    html: '',
    value: '{0}',
    data: '{}',
    sensitive_css_class: 'aCssClass',
    sort_table_ids: [],
  }),
});

// Custom View. Renders the widget model.
const SensitiveHTMLView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.sensitive = SensitiveHTML(this);
    this.html_changed();
    // Observe changes in the value traitlet in Python, and define
    // a custom callback.
    this.model.on('change:html', this.html_changed, this);
    this.model.on('change:data', this.data_changed, this);
  },

  html_changed: function () {
    this.el.innerHTML = this.model.get('html');
    let that = this;
    let sensitive_class = this.model.get('sensitive_css_class');
    elementReady('.' + sensitive_class).then(() => {
      that.sensitive.update_cb();
      let sort_table_ids = this.model.get('sort_table_ids');
      for (const i in sort_table_ids) {
        let tid = sort_table_ids[i];
        let tobj = document.getElementById(tid);
        sorttable.makeSortable(tobj);
      }
    });
  },

  data_changed: function () {
    let that = this;
    let sensitive_class = this.model.get('sensitive_css_class');
    elementReady('.' + sensitive_class).then(() =>
      that.sensitive.update_data()
    );
  },
});

const DataTableModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: 'DataTableModel',
    _view_name: 'DataTableView',
    _model_module: 'progressivis-nb-widgets',
    _view_module: 'progressivis-nb-widgets',
    _model_module_version: '0.1.0',
    _view_module_version: '0.1.0',
    columns: '[a, b, c]',
    data: 'Hello DataTable!',
    page: '{0}',
    dt_id: 'aDtId',
  }),
});

// Custom View. Renders the widget model.
const DataTableView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.id = 'datatable_' + this.model.get('dt_id') + new_id();
    this.data_table = null;
    this.data_changed();
    // Observe changes in the value traitlet in Python, and define
    // a custom callback.
    this.model.on('change:data', this.data_changed, this);
  },

  data_changed: function () {
    //const dt_id = this.model.get('dt_id');
    const dt_id = this.id;
    if (document.getElementById(this.id) == null) {
      this.el.innerHTML = `<div style='overflow-x:auto;'><table id='${dt_id}' class='display' style='width:100%;'></div>`;
    }
    let that = this;
    elementReady('#' + this.id).then(() => dt.update_table(that, dt_id));
  },
});

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
  ScatterplotModel: ScatterplotModel,
  ScatterplotView: ScatterplotView,
  ModuleGraphModel: ModuleGraphModel,
  ModuleGraphView: ModuleGraphView,
  SensitiveHTMLModel: SensitiveHTMLModel,
  SensitiveHTMLView: SensitiveHTMLView,
  JsonHTMLModel: JsonHTMLModel,
  JsonHTMLView: JsonHTMLView,
  SparkLineProgressBarModel: SparkLineProgressBarModel,
  SparkLineProgressBarView: SparkLineProgressBarView,
  PlottingProgressBarModel: PlottingProgressBarModel,
  PlottingProgressBarView: PlottingProgressBarView,
  DataTableModel: DataTableModel,
  DataTableView: DataTableView,
};
