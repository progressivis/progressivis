// Canvas props

var line_graph = (function() {

var margin = {top: 30, right: 20, bottom: 30, left: 50},
    width = 600 - margin.left - margin.right,
    height = 200 - margin.top - margin.bottom;


// ranges & axes

var x = d3.scaleLinear().range([0, width]);
var y = d3.scaleLinear().range([height, 0]);

var xAxis = d3.axisBottom().scale(x).ticks(5);
var yAxis = d3.axisLeft().scale(y).ticks(5);

// Define the line func

var lineval = d3.line()
    .x(function(d) { return x(d.step); })
    .y(function(d) { return y(d.quality); });
    

function line_graph(data){
    try{
        d3.select("#d3quality svg").remove();
    } catch(e) {};
    qsvg = d3.select("#d3quality").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform", 
              "translate(" + margin.left + "," + margin.top + ")");

    // reshape the data
    var idx = data.index;
    var qual_data = []; 
    for(i in idx){
        obj = {step: parseFloat(i),
               quality: data.quality[i]};
        qual_data.push(obj);
    }
    //console.log("qual: ", qual_data)
    x.domain(d3.extent(qual_data, function(d) { return d.step; }));
    y.domain(d3.extent(qual_data, function(d) { return d.quality; }));
    qsvg.append("path").style("fill","none")
        .attr("class", "line")
        .attr("d", lineval(qual_data));
    // X, Y Axis
    qsvg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    qsvg.append("g")
        .attr("class", "y axis")
        .call(yAxis);
    

}

return line_graph;
})();
