"use strict";

var colormaps = function(){
  const svgNS = "http://www.w3.org/2000/svg";
  
  var tables = {
    default: [[1,1,1],[0.93,0.001,0],[0.63,0.001,0]],
    reds: [[1,0.75,0.5],[0.75,0.3,0],[0.75,0.3,0]],
    blues: [[0.75,0.3,0],[0.75,0.3,0],[1,0.75,0.5]],
    grays: [[1,0],[1,0],[1,0]]
  };

  /**
   * @param xfer - a feComponentTransfer element that will be mutated.
   * @param name - a table name (use <pre>getTableNames</pre> to retrieve a list)
   */
  function makeTableFilter(xfer, name){
    if(!xfer){
      console.warn("makeTableFilter requires an existing feComponentTransfer element");
      return;
    }
    var table = tables[name];
    if(!table){
      console.warn("unknown table: " + name);
      return;
    }
    //remove all feComponentTransfer children
    while(xfer.firstChild){
      xfer.removeChild(xfer.firstChild);
    }
    var rx = document.createElementNS(svgNS, "feFuncR");
    rx.setAttribute("type", "table");
    rx.setAttribute("tableValues", tableString(table[0]));
    var gx = document.createElementNS(svgNS, "feFuncG");
    gx.setAttribute("type", "table");
    gx.setAttribute("tableValues", tableString(table[1]));
    var bx = document.createElementNS(svgNS, "feFuncB");
    bx.setAttribute("type", "table");
    bx.setAttribute("tableValues", tableString(table[2]));

    xfer.appendChild(rx);
    xfer.appendChild(gx);
    xfer.appendChild(bx);
  }

  function tableString(table){
    return table.reduce(function(acc, elt){
      return acc + elt + " ";
    }, "");
  }

  function getTableNames(){
    return Object.keys(tables);
  }

  return {
    makeTableFilter: makeTableFilter,
    getTableNames: getTableNames
  };
}();
