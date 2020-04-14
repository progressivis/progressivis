import {DataTable} from 'datatables';
import $ from 'jquery';

var data_table = null;

function change_page(wobj, data_table){
    var info = data_table.page.info();
    info['draw'] = data_table.context[0].oAjaxData.draw + 1;
    wobj.model.set("page", info);
    console.log("info:", info)
    wobj.touch();
}


function update_table(wobj, dt_id) {
    let data = wobj.model.get('data');
    let columns_ = JSON.parse(wobj.model.get('columns'));
    console.log(data)
   
    console.log("dt_id:", dt_id);
    if ( ! $.fn.DataTable.isDataTable( '#'+dt_id ) ) {
	console.log("Create DT:"+dt_id);
        var columns = columns_.map(function(c) { return {"sTitle": c.toString()}; });
        data_table = $( '#'+dt_id).DataTable( {
            "columns": columns,
            "processing": true,
            "serverSide": true,
            //"retrieve": true,
	    "ajax": function (data_, callback, settings) {
		var stuff = JSON.parse(wobj.model.get('data'));
		if(stuff.draw < data_.draw) stuff.draw = data_.draw;
		callback(stuff);
		
            }}).on( 'page.dt', function () {
		change_page(wobj, data_table);
	    } ).on( 'length.dt', function ( e, settings, len ) {
		change_page(wobj, data_table);
	    });//window.my_table = data_table;
    } else {
        //$('#dataframe').dataTable({"retrieve": true}).ajax.reload();
        data_table.ajax.reload(null, false);
    }
}



export {update_table}
