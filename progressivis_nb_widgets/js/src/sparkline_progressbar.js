import $ from 'jquery';
import 'jquery-sparkline';



function update_slpb(view_){
    let data = view_.model.get('data');
    let values = data.values;
    let progress = data.progress;
    let type_ = data.type||'line';    
    $('#sparkline-pb').css('width', progress+'%');
    $('#sparkline-pb').sparkline(values, {type: type_, height: '100%', width: '100%'});
}
export {update_slpb};
