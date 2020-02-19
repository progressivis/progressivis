var widgets = require('@jupyter-widgets/base');
var _ = require('lodash');
html_  = require('./sc_template');
mc2d = require('./multiclass2d');
er = require('./es6-element-ready');
mg = require('./module_graph');
require('./module-graph.css');
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
        data : 'Hello Scatterplot!',
	value: '{0}'
    })
});


// Custom View. Renders the widget model.
var ScatterplotView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {
	this.el.innerHTML = html_;
	let that = this;	
	er.elementReady("#prevImages").then((_)=>{
	    mc2d.ready_(that);
	});
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:data', this.data_changed, this);

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
	this.el.innerHTML = '<svg id="module-graph" width="960" height="500"></svg>';      var that = this;
	er.elementReady("#module-graph").then((_)=>{
	    mg.graph_setup();
	    that.data_changed();
	});
	console.log("REnder ModuleGraphView");
        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:data', this.data_changed, this);
	console.log("this.model done", this.model); 

    },

    data_changed: function() {
	console.log("Data changed ModuleGraphView");
	let val = this.model.get('data');
	mg.graph_update(JSON.parse(val));	
	
    }
});


module.exports = {
    ScatterplotModel: ScatterplotModel,
    ScatterplotView: ScatterplotView,
    ModuleGraphModel: ModuleGraphModel,
    ModuleGraphView: ModuleGraphView
    };