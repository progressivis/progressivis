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
      acc.push(histogram1d.process_histogram(histograms[key]));
      return acc;
    }, []);

    var charts = d3hists.map(function(elt){
      return histogram1d.chart(elt);
    });

    var chartElts = d3.select("#histograms").selectAll(".hist").data(charts);
    chartElts.enter()
             .append("div")
             .attr("class", "hist");
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
