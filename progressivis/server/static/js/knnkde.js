function translate(x, y) { return 'translate(' + x + ',' + y + ')'; }

function DensityPlot(svgId) {
  let width = 380, height = 380;
  let svg = d3.select(svgId).append('g').attr('transform', translate(10, 10))
  let bg = svg.append('g')
  let fg = svg.append('g')
  let step = 10

  return function(data) {
    data.points.forEach(d => {
      d[0] = +d[0] // x
      d[1] = +d[1] // y
    })

    data.samples.forEach(d => {
      d[0][0] = +d[0][0] // x
      d[0][1] = +d[0][1] // y
      d[1] = +d[1] // density
    })

    let x = d3.scaleLinear().domain(d3.extent(data.samples, d => d[0][0])).range([0, width])
    let y = d3.scaleLinear().domain(d3.extent(data.samples, d => d[0][1])).range([0, height])

    fg
      .selectAll('circle')
      .data(data.points)
      .enter()
      .append('circle')
        .attr('r', 3)
        .attr('fill', 'white')
        .attr('stroke', 'black')
        .attr('stroke-width', '2px')
        .attr('cx', d => x(d[0]))
        .attr('cy', d => y(d[1]))
        .attr('opacity', 0.7)
    
    let bins = +data.bins;
    let color = d3.scaleSequential(d3.interpolateCool)
      .domain([0, d3.max(data.samples, d => d[1])])
    let densityMax = d3.max(data.samples, x => x[1])

    let paths = bg.selectAll('path')
      .data(d3.contours()
        .size([bins + 1, bins + 1])
        .thresholds(d3.range(0, densityMax, densityMax / step))
        (data.samples.map(d => d[1]))
      )

    let enter = paths
      .enter()
        .append('path')

    paths.merge(enter)
      .attr('d', d3.geoPath(d3.geoTransform({
        point: function(x, y) {
          let s = width / bins;
          this.stream.point(y * s - s * 0.5, x * s - s * 0.5)
        }
      }))) 
      .attr('fill', d => color(d.value))
  }
}



function knnkde_refresh(json) {
    if(json && json.payload) {
        knnkde_update(json.payload);}
    else {
        module_set_hotline(true);
        module_get(knnkde_update, error);
    }    
}


function knnkde_update_vis(data) {
    densityPlot(data);
}
function knnkde_update(data) {
    module_update(data);
    knnkde_update_vis(data);
}


function knnkde_ready() {
    densityPlot = DensityPlot('#knn')
    module_ready();
    refresh = knnkde_refresh;
}
