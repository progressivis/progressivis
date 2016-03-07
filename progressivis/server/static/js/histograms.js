var histograms = function() {

  var hists = [];

  function ready(){
    refresh = histograms_refresh;
    module_ready();
  }

  function histograms_refresh() {
    module_get(update, error);
  }

  function update(data) {
    module_update(data);

    var histograms = data.histograms;
    var d3hists = Object.keys(histograms).reduce(function(acc, key){
      var d3hist = histogram1d.process_histogram(histograms[key]);
      d3hist.columnName = key;
      acc.push(d3hist);
      return acc;
    }, []);

    var charts = d3hists.map(function(elt){
      return histogram1d.chart(elt);
    });

    var histElts = d3.select("#histograms").selectAll(".hist").data(d3hists);
    var hist = histElts.enter()
                      .append("div")
                      .attr("class", "hist");
    hist.append("div")
        .attr("class", "colName")
        .text(function(d){ return d.columnName; });
    hist.append("div")
        .attr("class", "chart");

    var chartElts = d3.select("#histograms").selectAll(".chart").data(charts);
    
    chartElts.each(function(elt, idx){
      //update data
      charts[idx].histogram(d3hists[idx]);
      //render
      charts[idx](d3.select(this));
    });

  }

  function error(err) {
    console.log(err);
  }

  return {
    ready: ready,
  };
}();
