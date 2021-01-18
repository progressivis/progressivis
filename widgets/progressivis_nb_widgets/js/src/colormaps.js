'use strict';

/**
 * Color function transfer management.
 * Requires chroma.js
 */
const chroma = require('chroma-js');
const svgNS = 'http://www.w3.org/2000/svg';

const tables = {
  Default: [
    [0, 1],
    [0, 1],
    [0, 1],
  ], //[[1,1,1],[0.93,0.001,0],[0.63,0.001,0]],
  OrRd: tableFromChroma(chroma.brewer.OrRd),
  PuBu: tableFromChroma(chroma.brewer.PuBu),
  Oranges: tableFromChroma(chroma.brewer.Oranges),
  Greens: tableFromChroma(chroma.brewer.Greens),
  CubeHelix: tableFromChroma(
    chroma.cubehelix().scale().correctLightness().colors(9)
  ),
  InvertedGrayscale: [
    [1, 0.01, 0],
    [1, 0.01, 0],
    [1, 0.01, 0],
  ],
  Identity: [
    [0, 1],
    [0, 1],
    [0, 1],
  ],
};

function tableFromChroma(chromaTable) {
  const rtab = [];
  const gtab = [];
  const btab = [];

  chromaTable.forEach((chromaElt) => {
    var rgb = chroma(chromaElt).get('rgb');
    rtab.push(rgb[0] / 255);
    gtab.push(rgb[1] / 255);
    btab.push(rgb[2] / 255);
  });

  return [rtab, gtab, btab];
}

function createAndAddTransferNode(parent, name, tableValues) {
  const node = document.createElementNS(svgNS, name);
  node.setAttribute('type', 'table');
  node.setAttribute('tableValues', tableString(tableValues));
  parent.appendChild(node);
}

/**
 * @param xfer - a feComponentTransfer element that will be mutated.
 * @param name - a table name (use <pre>getTableNames</pre> to retrieve a list)
 */
export function makeTableFilter(xfer, name) {
  if (!xfer) {
    console.warn(
      'makeTableFilter requires an existing feComponentTransfer element'
    );
    return null;
  }
  const table = tables[name];
  if (!table) {
    console.warn('unknown table: ' + name);
    return null;
  }
  //remove all feComponentTransfer children
  while (xfer.firstChild) {
    xfer.removeChild(xfer.firstChild);
  }
  createAndAddTransferNode(xfer, 'feFuncR', table[0]);
  createAndAddTransferNode(xfer, 'feFuncG', table[1]);
  createAndAddTransferNode(xfer, 'feFuncB', table[2]);

  return xfer;
}

function tableString(table) {
  return table.reduce((acc, elt) => acc + elt + ' ', '');
}

export function getTableNames() {
  return Object.keys(tables);
}
