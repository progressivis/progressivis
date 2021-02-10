import 'datatables';
import $ from 'jquery';

function change_page(wobj) {
  const info = wobj.data_table.page.info();
  info.draw = wobj.data_table.context[0].oAjaxData.draw + 1;
  wobj.model.set('page', info);
  wobj.touch();
}

export function update_table(wobj, dt_id) {
  const cols = wobj.model.get('columns');
  if (cols == '') return;
  const columns_ = JSON.parse(cols);
  //console.log(data)
  if (!wobj.data_table) {
    const columns = columns_.map((c) => ({ sTitle: c.toString() }));
    wobj.data_table = $('#' + dt_id)
      .DataTable({
        columns: columns,
        processing: true,
        serverSide: true,
        //"retrieve": true,
        ajax: (data_, callback) => {
          const js_data = JSON.parse(wobj.model.get('data'));
          if (js_data.draw < data_.draw) js_data.draw = data_.draw;
          callback(js_data);
        },
      })
      .on('page.dt', () => change_page(wobj))
      .on('length.dt', () => change_page(wobj));
  } else {
    wobj.data_table.ajax.reload(null, false);
  }
}
