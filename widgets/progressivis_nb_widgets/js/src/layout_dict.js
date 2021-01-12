import $ from 'jquery';
import 'jquery-sparkline';
import { elementReady } from "./es6-element-ready";

var ipyView = null;
var dom_id = null;

function makeSparkId(k) {
  return 'ps-spark_'+dom_id+'_'+k;
}

function escapeHTML(s) {
  return $('<div>').text(s).html();
}

function layout_value(v) {
  var i, layout = "";
  if (v == null) return "";
  if (Array.isArray(v)) {
    if (v.length === 0) return "";
    for (i = 0; i < v.length; i++) {
      if (layout.length != 0)
	layout += "<br>";
      layout += layout_value(v[i]);
    }
    return layout;
  }
  if (typeof(v) === "string" && v.startsWith('<div')) {
    return v;
  }
  if (typeof(v) === "object") {
    var keys = Object.keys(v);
    if (keys.length == 0) return "";
    return layout_dict(v, keys.sort());
  }
  return escapeHTML(v.toString()); // escape
}


function sparkline_disp(v, k) {
  var SIZE = 15;
  var last = v[v.length-1];
  last = last.toFixed(0);
  last = last+"&nbsp;".repeat(SIZE-last.length);
  return `<table>
    <tr><td>${last}</td><td><span class='ps-sparkline' id='${makeSparkId(k)}'>...</span></td></tr>
  </table>`;
}

function layout_dict(dict, order, value_func={}) {
  var i, k, v, layout = '';

  if (! order)
    order = Object.keys(dict).sort();
  layout += '<dl class="dl-horizontal">';
  for (i = 0; i < order.length; i++) {
    k = order[i];
    v = dict[k];
    layout += ' <dt>'+k.toString()+':</dt>';
    layout += ' <dd>';
    if(value_func[k]){
      layout += value_func[k](v, k);            
    } else {
      layout += layout_value(v);
    }
    layout += '</dd>';
  }
  layout += '</dl>';
  return layout;
}

function layout_dict_entry(view_) {
  ipyView = view_;
  dom_id = ipyView.model.get('dom_id');
  let  jq_id = '#'+ipyView.model.get('dom_id');
  let data = view_.model.get('data');
  let config = view_.model.get('config');     
  let order = config.order;
  let sparkl = config.sparkline||[];
  let procs = {};
  for(const k of sparkl){
    procs[k] = sparkline_disp;
  }
  $(jq_id).html(layout_dict(data, order, procs));
  elementReady('.ps-sparkline').then(()=>{
    for(const k of sparkl){
      $('#'+makeSparkId(k)).sparkline(data[k]);
    }
  });
  
}
export {layout_dict_entry};
