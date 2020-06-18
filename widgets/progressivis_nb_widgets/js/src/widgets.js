"use strict";
var widgets = require('@jupyter-widgets/base');
var _ = require('lodash');
var html_  = require('./sc_template');
var mc2d = require('./multiclass2d');
var er = require('./es6-element-ready');
var mg = require('./module_graph');
var dt = require('./data_table');
var slpb = require('./sparkline_progressbar');
var lg = require('./line_graph');
require('../css/module-graph.css');
require('../css/sparkline-progressbar.css');
require('datatables/media/css/jquery.dataTables.css');
var sch = require('./sensitive_html');
var jh = require('./layout_dict');

import {
    ISerializers, data_union_serialization, getArray,
    listenToUnion
} from 'jupyter-dataserializers';

var ndarray = require('ndarray');
window.ndarray = ndarray;
// See example.py for the kernel counterpart to this file.


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
var ScatterplotModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'ScatterplotModel',
        _view_name : 'ScatterplotView',
        _model_module : 'progressivis-nb-widgets',
        _view_module : 'progressivis-nb-widgets',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
	h1: ndarray([]),
        data : 'Hello Scatterplot!',
	value: '{0}',
	move_point: '{0}'	
    })
    
}, {
    serializers: _.extend({
        h1: data_union_serialization,
    }, widgets.DOMWidgetModel.serializers),
});
/*
ScatterplotModel.serializers =  _.extend({
    h1: {
        serialize: array_serialization
    }
}, widgets.DOMWidgetModel.prototype.serializers)
*/

// Custom View. Renders the widget model.
var ScatterplotView = widgets.DOMWidgetView.extend({
    /*initialize: function(parameters) {
	widgets.DOMWidgetView.prototype.initialize(parameters);
	listenToUnion(this.model, 'h1', this.update.bind(this), true);
    },*/

    // Defines how the widget gets rendered into the DOM
    render: function() {
	this.el.innerHTML = html_;
	let that = this;	
	er.elementReady("#prevImages").then((_)=>{
	    mc2d.ready_(that);
	    });
	listenToUnion(this.model, 'h1', this.update.bind(this), true);
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:data', this.data_changed, this);
        //this.model.on('change:h1', this.h1_changed, this);	

    },
    h1_changed: function(){
	// only for debugging purposes
	console.log("h1_changed");
	console.log("h1", this.model.get('h1'));
	
    },
    data_changed: function() {
	let val = this.model.get('data');
	mc2d.update_vis(JSON.parse(val));
    }
});


// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
var ModuleGraphModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'ModuleGraphModel',
        _view_name : 'ModuleGraphView',
        _model_module : 'progressivis-nb-widgets',
        _view_module : 'progressivis-nb-widgets',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        data : 'Hello ModuleGraph!'
	//value: '{0}'
    })
});


// Custom View. Renders the widget model.
var ModuleGraphView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {
	this.el.innerHTML = '<svg id="module-graph" width="960" height="500"></svg>';
	var that = this;
	er.elementReady("#module-graph").then((_)=>{
	    mg.graph_setup();
	    that.data_changed();
	});
	console.log("REnder ModuleGraphView");
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:data', this.data_changed, this);

    },

    data_changed: function() {
	console.log("Data changed ModuleGraphView");
	let val = this.model.get('data');
	if(val=='{}') return;
	mg.graph_update(JSON.parse(val));	
	
    }
});

var SensitiveHTMLModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'SensitiveHTMLModel',
        _view_name : 'SensitiveHTMLView',
        _model_module : 'progressivis-nb-widgets',
        _view_module : 'progressivis-nb-widgets',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        html : '',
	value: '{0}',
	data: '{}',	
	sensitive_css_class: 'aCssClass'
    })
});


// Custom View. Renders the widget model.
var SensitiveHTMLView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {
	this.html_changed();
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:html', this.html_changed, this);
        this.model.on('change:data', this.data_changed, this);
    },

    html_changed: function() {
	this.el.innerHTML = this.model.get('html');
	let that = this;
	let sensitive = this.model.get('sensitive_css_class');
	er.elementReady('.'+sensitive).then((_)=>{
	    sch.update_cb(that);
	});
    },
    
    data_changed: function() {
	let that = this;
	let sensitive = this.model.get('sensitive_css_class');
	er.elementReady('.'+sensitive).then((_)=>{
	    sch.update_data(that);
	});
    }
});

var DataTableModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'DataTableModel',
        _view_name : 'DataTableView',
        _model_module : 'progressivis-nb-widgets',
        _view_module : 'progressivis-nb-widgets',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        columns: '[a, b, c]',
        data: 'Hello DataTable!',	
	page: '{0}',
	dt_id: 'aDtId'
    })
});


// Custom View. Renders the widget model.
var DataTableView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {
	this.data_changed();
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:data', this.data_changed, this);

    },

    data_changed: function() {
	let dt_id = this.model.get('dt_id');
	if(document.getElementById(dt_id)==null){
	    this.el.innerHTML =
		"<div style='overflow-x:auto;'><table id='"+dt_id+
		"' class='display' style='width:100%;'></div>";
	}
	let that = this;
	er.elementReady('#'+dt_id).then((_)=>{
	    dt.update_table(that, dt_id);
	});
    }
});

var JsonHTMLModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'JsonHTMLModel',
        _view_name : 'JsonHTMLView',
        _model_module : 'progressivis-nb-widgets',
        _view_module : 'progressivis-nb-widgets',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        dom_id : 'json_html_dom_id',
	data: '{}',
	config: '{}'
    })
});


// Custom View. Renders the widget model.
var JsonHTMLView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {
        let dom_id = this.model.get('dom_id');
        this.el.innerHTML = "<div id='"+dom_id+"'></div>";
	this.data_changed();
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:config', this.data_changed, this);
        this.model.on('change:data', this.data_changed, this);
    },

    
    data_changed: function() {
	let that = this;
        let dom_id = this.model.get('dom_id');
   
	er.elementReady('#'+dom_id).then((_)=>{
	    jh.layout_dict_entry(that);
	});
    }
});

var SparkLineProgressBarModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'SparkLineProgressBarModel',
        _view_name : 'SparkLineProgressBarView',
        _model_module : 'progressivis-nb-widgets',
        _view_module : 'progressivis-nb-widgets',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
	data: '{}'
    })
});

// Custom View. Renders the widget model.
var SparkLineProgressBarView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {

        this.el.innerHTML = `<div style="width: 100%;">
			<div class="slpb-bg">
				<span id='sparkline-pb' class="slpb-fill" style="width: 70%;"></span>
			</div>
		</div>`;
	this.data_changed();
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:data', this.data_changed, this);
    },

    
    data_changed: function() {
	let that = this;   
	er.elementReady('#sparkline-pb').then((_)=>{
	    slpb.update_slpb(that);
	});
    }
});

var SparkLineProgressBarModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'SparkLineProgressBarModel',
        _view_name : 'SparkLineProgressBarView',
        _model_module : 'progressivis-nb-widgets',
        _view_module : 'progressivis-nb-widgets',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
	data: '{}'
    })
});

// Custom View. Renders the widget model.
var SparkLineProgressBarView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {

        this.el.innerHTML = `<div style="width: 100%;">
			<div class="slpb-bg">
				<span id='sparkline-pb' class="slpb-fill" style="width: 70%;"></span>
			</div>
		</div>`;
	this.data_changed();
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:data', this.data_changed, this);
    },

    
    data_changed: function() {
	let that = this;   
	er.elementReady('#sparkline-pb').then((_)=>{
	    slpb.update_slpb(that);
	});
    }
});


var PlottingProgressBarModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'PlottingProgressBarModel',
        _view_name : 'PlottingProgressBarView',
        _model_module : 'progressivis-nb-widgets',
        _view_module : 'progressivis-nb-widgets',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
	data: '{}'
    })
});

// Custom View. Renders the widget model.
var PlottingProgressBarView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {

        this.el.innerHTML = `<div style="width: 100%;">
			<div class="slpb-bg">
				<div id='plotting-pb' class="slpb-fill" style="width: 70%;"></div>
			</div>
		</div>`;
	this.data_changed();
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:data', this.data_changed, this);
    },

    
    data_changed: function() {
	let that = this;   
	er.elementReady('#plotting-pb').then((_)=>{
	    slpb.update_slpb(that);
	});
    }
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
    DataTableView: DataTableView
    };
