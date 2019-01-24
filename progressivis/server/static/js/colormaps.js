"use strict";

/**
 * Color function transfer management.
 * Requires chroma.js
 */
var colormaps = function(){
  const svgNS = "http://www.w3.org/2000/svg";
  
  var tables = {
    Default: [[0,1],[0,1],[0,1]], //[[1,1,1],[0.93,0.001,0],[0.63,0.001,0]],
    OrRd: tableFromChroma(chroma.brewer.OrRd),
    PuBu: tableFromChroma(chroma.brewer.PuBu),
    Oranges: tableFromChroma(chroma.brewer.Oranges),
    Greens:  tableFromChroma(chroma.brewer.Greens),
    CubeHelix: tableFromChroma(chroma.cubehelix().scale().correctLightness().colors(9)),
    InvertedGrayscale: [[1,0.01,0],[1,0.01,0],[1,0.01,0]],
    Identity: [[0,1],[0,1],[0,1]]
  };

  function tableFromChroma(chromaTable){
    var rtab = [];
    var gtab = [];
    var btab = [];
    
    chromaTable.forEach(function(chromaElt){
      var rgb = chroma(chromaElt).get('rgb');
      rtab.push(rgb[0]/255);
      gtab.push(rgb[1]/255);
      btab.push(rgb[2]/255);
    });

    return [rtab, gtab, btab];
  }

  function createAndAddTransferNode(parent, name, tableValues){
    var node =  document.createElementNS(svgNS, name);
    node.setAttribute("type", "table");
    node.setAttribute("tableValues", tableString(tableValues));
    parent.appendChild(node);
  }
  /**
   * @param xfer - a feComponentTransfer element that will be mutated.
   * @param name - a table name (use <pre>getTableNames</pre> to retrieve a list)
   */
  function makeTableFilter(xfer, name){
    if(!xfer){
      console.warn("makeTableFilter requires an existing feComponentTransfer element");
      return null;
    }
    var table = tables[name];
    if(!table){
      console.warn("unknown table: " + name);
      return null;
    }
    //remove all feComponentTransfer children
    while(xfer.firstChild){
      xfer.removeChild(xfer.firstChild);
    }
    createAndAddTransferNode(xfer, "feFuncR", table[0]);
    createAndAddTransferNode(xfer, "feFuncG", table[1]);
    createAndAddTransferNode(xfer, "feFuncB", table[2]);

    return xfer;
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
