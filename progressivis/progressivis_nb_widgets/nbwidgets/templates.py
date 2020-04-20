from io import StringIO

index_tpl = """<table class="table table-striped table-bordered table-hover table-condensed">
	  <thead>
	    <tr><th>Id</th><th>Class</th><th>State</th><th>Last Update</th><th>Order</th></tr>
	  </thead>
	  <tbody>
          {% for m in modules%}
          <tr>
          {% for c in cols%}
          <td>
          {% if c!='id' %}
          <span id="ps-cell_{{m['id']}}_{{c}}">{{m[c]}}</span>
          {% else %}
          <!--button class='ps-row-btn' id="ps-row-btn_{{m[c]}}" type='button' >{{m[c]}}</button-->
          <a class='ps-row-btn' id="ps-row-btn_{{m[c]}}" type='button' >{{m[c]}}</a>
          {% endif %}
          </td>
          {%endfor %}
          </tr>
          {%endfor %}
	  </tbody>
        </table>"""



def layout_value(v, layout):
    if not v:
        return
    if isinstance(v, list):
        for e in v:
            layout_value(e, layout)
    elif isinstance(v, str) and v.startswith('<div'):
        return
    if isinstance(v, dict):
        layout_dict(v, sorted(v.keys()), layout=layout)
    else:
        layout.write(f'<div>{v}</div>')



def layout_dict(data, order=None, layout=None, value_func={}):
    if layout is None:
        layout = StringIO()
    if order is None:
        order = sorted(data.keys())
    layout.write('<dl class="dl-horizontal">')
    for k in order:
        if k not in data:
            continue
        layout.write(f' <dt>{k}:</dt><dd>')        
        v = data[k];
        if k in value_func:
            value_func[k](v, layout)
        else:
            layout_value(v, layout)
        layout.write('</dd>')
    layout.write('</dl>');
    return layout.getvalue()

