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

