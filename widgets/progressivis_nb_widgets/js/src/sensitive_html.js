import $ from 'jquery';

export function SensitiveHTML(ipyView) {
  function onclick_fun() {
    console.log('click on row:', this.id);
    ipyView.model.set('value', this.id);
    ipyView.touch();
  }

  function update_cb() {
    let cssCls = ipyView.model.get('sensitive_css_class');
    $(`#${ipyView.id} .${cssCls}`)
      .unbind('click')
      .click(onclick_fun);
  }
  function update_data() {
    let data = ipyView.model.get('data');
    let k = null;
    for (k in data) {
      $('#' + k).html(data[k]);
    }
  }
  return { update_cb, update_data };
}
