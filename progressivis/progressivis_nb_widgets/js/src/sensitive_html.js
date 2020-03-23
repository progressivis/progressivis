import $ from 'jquery';
var ipyView = null;
function onclick_fun(){
    ipyView.model.set('value', this.id);
    ipyView.touch();
}
function update_cb(view_){
    ipyView = view_;
    let cssCls = ipyView.model.get('sensitive_css_class');
    $('.'+cssCls).unbind('click').click(onclick_fun);
}
export  {update_cb};
