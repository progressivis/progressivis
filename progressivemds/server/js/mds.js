function mds(error, data) {
	data.forEach(function(d) {
		d.x = +d.x;
		d.y = +d.y;
		d.color = d3.rgb(d.color);
    });

    x.domain(d3.extent(data, function(d) { return d.x; })).nice();
    y.domain(d3.extent(data, function(d) { return d.y; })).nice();

    svg.append("g")
	.attr("class", "x axis")
	.attr("transform", "translate(0," + height + ")")
	//.call(xAxis.tickSize(-height, 0, 0))
	.call(xAxis)
	.append("text")
	.attr("class", "label")
	.attr("x", width)
	.attr("y", -6)
	.style("text-anchor", "end")
	.text("No Meaning");

    svg.append("g")
	.attr("class", "y axis")
	//.call(yAxis.tickSize(-width, 0, 0))
	.call(yAxis)
	.append("text")
	.attr("class", "label")
	.attr("transform", "rotate(-90)")
	.attr("y", 6)
	.attr("dy", ".71em")
	.style("text-anchor", "end")
	.text("No Meaning");

    svg.selectAll(".dot")
	.data(data, function(d) { return d.id; })
	.enter().append("circle")
	.attr("class", "dot")
	.attr("r", 3.5)
	.attr("cx", function(d) { return x(d.x); })
	.attr("cy", function(d) { return y(d.y); })
	.style("fill", function(d) { return d.color; })
	.append("title").text(function(d) { return d.label; });
};

function update_mds(error, data) {
	data.forEach(function(d) {
		d.x = +d.x;
		d.y = +d.y;
		d.color = d3.rgb(d.color);
    });
	
	svg.select(".x.axis")
		.transition().duration(500)
		.call(xAxis);
	
	svg.select(".y.axis")
		.transition().duration(500)
		.call(yAxis);
	
	var dots = svg.selectAll(".dot")
		.data(data, function(d) { return d.id; });
	dots.enter().append("circle")
		.attr("class", "dot")
		.attr("r", 3.5)
		.attr("cx", function(d) { return x(d.x); })
		.attr("cy", function(d) { return y(d.y); })
		.style("fill", function(d) { return d.color; })
		.append("title").text(function(d) { return d.label; });
	dots.exit()
		.attr("class", "exit")
		.transition()
			.duration(750)
			.style("fill-opacity", 1e-6)
			.remove();
	dots.transition() // update
    	.duration(750)
    	.attr("cx", function(d) { return x(d.x); })
    	.attr("cy", function(d) { return y(d.y); })
    	.style("fill", function(d) { return d.color; });
}