function heatmapSetData(hmObj, data, xbins, ybins, min=0, max=255) {
      // reset data arrays
      hmObj._data = [];
      hmObj._radi = [];
      var xLen = xbins; //data.length;
      var yLen = ybins; //data[0].length;
      for (var i=0; i<xLen;i++){
          for(var j=0; j<yLen;j++){
             v = data[i][j];
             if(!v) continue;
             hmObj._store._organiseData({x: j, y: xLen-i-1,
                                   value: v}, false);
          }
      }
   
      hmObj._max = max;
      hmObj._min = min;
      
      hmObj._store._onExtremaChange();
      hmObj._store._coordinator.emit('renderall', hmObj._store._getInternalData());
      return hmObj;
}
