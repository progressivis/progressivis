import $ from 'jquery';
let ipyView = null;

function onclick_fun() {
  console.log('click on row:', this.id);
  ipyView.model.set('value', this.id);
  ipyView.touch();
}
function update_cb(view_) {
  ipyView = view_;
  let cssCls = ipyView.model.get('sensitive_css_class');
  $('.' + cssCls)
    .unbind('click')
    .click(onclick_fun);
}

function update_data(view_) {
  ipyView = view_;
  let data = ipyView.model.get('data');
  let k = null;
  for (k in data) {
    $('#' + k).html(data[k]);
  }
}
export { update_cb, update_data };
