//"use strict";
// Export widget models and views, and the npm package version number.
require("./config-editor");
//split config-editor because babel seems ignore jquery import when react is present
require("./config-editor-disp");
module.exports = require('./widgets.js');
module.exports['version'] = require('../package.json').version;

