import 'datatables';
import $ from 'jquery';


function change_page(wobj){
    const info = wobj.data_table.page.info();
    info['draw'] = wobj.data_table.context[0].oAjaxData.draw + 1;
    wobj.model.set("page", info);
    //console.log("info:", info)
    wobj.touch();
}


export function update_table(wobj, dt_id) {
    //console.log("dt_id:", dt_id);
    //let data = wobj.model.get('data');
    const cols = wobj.model.get('columns');
    if(cols=="") return;
    //console.log("cols:", cols)
    const columns_ = JSON.parse(cols);
    //console.log(data)
    if (! wobj.data_table) { // ! $.fn.DataTable.isDataTable( '#'+dt_id ) ) {
	//console.log("Create DT:"+dt_id);
        const columns = columns_.map(function(c) { return {"sTitle": c.toString()}; });
        wobj.data_table = $('#'+dt_id).DataTable({
            "columns": columns,
            "processing": true,
            "serverSide": true,
            //"retrieve": true,
            "ajax": (data_, callback) => {
		const js_data = JSON.parse(wobj.model.get('data'));
		if(js_data.draw < data_.draw) js_data.draw = data_.draw;
		callback(js_data);
            }})
            .on('page.dt', () => change_page(wobj))
            .on('length.dt', () => change_page(wobj));
        //window.my_table = data_table;
    } else {
        //$('#dataframe').dataTable({"retrieve": true}).ajax.reload();
        wobj.data_table.ajax.reload(null, false);
    }
}

