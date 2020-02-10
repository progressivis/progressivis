//import ConfigForm from "./config-editor";
_ = require("./config-editor");
// Export widget models and views, and the npm package version number.
module.exports = require('./scatterplot.js');
module.exports['version'] = require('../package.json').version;
