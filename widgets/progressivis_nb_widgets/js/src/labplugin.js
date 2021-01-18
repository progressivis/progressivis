var plugin = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
  id: 'progressivis-nb-widgets',
  requires: [base.IJupyterWidgetRegistry],
  activate: (app, widgets) => {
    widgets.registerWidget({
      name: 'progressivis-nb-widgets',
      version: plugin.version,
      exports: plugin,
    });
  },
  autoStart: true,
};
