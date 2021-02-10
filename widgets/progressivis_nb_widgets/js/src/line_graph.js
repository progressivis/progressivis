//import * as d3 from 'd3';
import $ from 'jquery';

function update_pb(view_) {
  const id = view_.id;
  const data = view_.model.get('data');
  const values = data.values;
  const progress = data.progress;
  const type_ = data.type || 'line';

  $('#' + id).css('width', progress + '%');
  $('#' + id).sparkline(values, { type: type_, height: '100%', width: '100%' });
}
export { update_pb };
