import {DataTable} from 'datatables';
import $ from 'jquery';



function change_page(wobj){
    var info = wobj.data_table.page.info();
    info['draw'] = wobj.data_table.context[0].oAjaxData.draw + 1;
    wobj.model.set("page", info);
    //console.log("info:", info)
    wobj.touch();
}


function update_table(wobj, dt_id) {
    //console.log("dt_id:", dt_id);
    //let data = wobj.model.get('data');
    let cols = wobj.model.get('columns');
    if(cols=="") return;
    //console.log("cols:", cols)
    let columns_ = JSON.parse(cols);
    //console.log(data)
    if ( ! $.fn.DataTable.isDataTable( '#'+dt_id ) ) {
	//console.log("Create DT:"+dt_id);
        var columns = columns_.map(function(c) { return {"sTitle": c.toString()}; });
        wobj.data_table = $( '#'+dt_id).DataTable( {
            "columns": columns,
            "processing": true,
            "serverSide": true,
            //"retrieve": true,
	    "ajax": function (data_, callback, settings) {
		var js_data = JSON.parse(wobj.model.get('data'));
		if(js_data.draw < data_.draw) js_data.draw = data_.draw;
		callback(js_data);
		
            }}).on( 'page.dt', function () {
		change_page(wobj);
	    } ).on( 'length.dt', function ( e, settings, len ) {
		change_page(wobj);
	    });//window.my_table = data_table;
    } else {
        //$('#dataframe').dataTable({"retrieve": true}).ajax.reload();
        wobj.data_table.ajax.reload(null, false);
    }
}



export {update_table}
