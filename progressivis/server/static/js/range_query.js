var range_query = function(){

  function ready(){
    refresh = range_query_refresh;
    module_ready();
  }
  
  function range_query_refresh(){
    module_get(update, error);
  }

  function update(data){
    module_update(data);

    if(data.ranges === undefined){ return; }

    //render sliders
    var containers = d3.select("#sliders").selectAll(".slider").data(data.ranges);
    var cEnter = containers.enter()
                           .append("div")
                           .attr("class", "slider");

    cEnter.append("div")
          .attr("class", "slider-container");

    cEnter.append("div")
          .attr("class", "slider-name")
          .text(function(d){ return d.name; })
          
    //containers.call(d3.slider().value([10,40]).orientation("vertical"));
    //the line above is simpler but does not work -- maybe because of a bug in d3-sliders. All thumbs are mixed up.
    cEnter.selectAll(".slider-container")
              .each(function(d,i){ 
                      d3.select(this)
                        .call(d3.slider()
                                .value([d.out_min,d.out_max])
                                .axis(true)
                                .min(d.in_min)
                                .max(d.in_max)
                                .orientation("vertical")
                                .on('slideend', function(evt, value){
                                  //set module inputs
                                  var min = {};
                                  min[d.name] = value[0];
                                  var max = {};
                                  max[d.name] = value[1];
                                  module_input(min, update_success, error, module_id+"/min_value");
                                  module_input(max, update_success, error, module_id+"/max_value");
                                })); 
              });
  }

  function update_success(){}

  function error(err){
    console.error(err);
  }

  return {
    ready: ready
  };
}();
