from io import StringIO

index_tpl = """
<table id="mysortedtable" class="table table-striped table-bordered table-hover table-condensed">
<thead><tr><th></th><th>Id</th><th>Class</th><th>State</th><th>Last Update</th><th>Order</th></tr></thead>
<tbody>
{% for m in modules%}
  <tr>
  {% for c in cols%}
  <td>
  {% if c=='id' %}
  <a class='ps-row-btn' id="ps-row-btn_{{m[c]}}" type='button' >{{m[c]}}</a>
  {% elif c=='is_visualization' %}
  <span id="ps-cell_{{m['id']}}_{{c}}">{{'a' if m[c] else ' '}}</span>
  {% else %}
  <span id="ps-cell_{{m['id']}}_{{c}}">{{m[c]}}</span>
  {% endif %}
  </td>
  {%endfor %}
  </tr>
{%endfor %}
</tbody>
</table>
"""
