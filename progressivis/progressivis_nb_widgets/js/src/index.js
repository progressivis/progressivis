//"use strict";
// Export widget models and views, and the npm package version number.
require("./config-editor");
module.exports = require('./widgets.js');
module.exports['version'] = require('../package.json').version;

