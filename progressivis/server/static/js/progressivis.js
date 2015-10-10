var ProgressiVis = {};

var socket = null,
    handshake = false;

function websocket_open(msg, handler) {
    socket = new WebSocket("ws://" + document.domain + ":5000/websocket/");

    socket.onopen = function() {
        socket.send("ping "+msg);
	handshake = false;
    };

    socket.onmessage = function(message) {
        var txt = message.data;
	if (txt == 'pong' && !handshake) {
	    handshake = true;
	    //console.log('Received handshake');
	    return;
	}
	if (handler) handler(message);
    };
}

function websocket_submit(text) {
    socket.send(text);
}

function layout_dict(dict, order) {
    var i, k, v, layout = '';

    if (! order)
	order = Object.keys(dict).sort();
    layout += '<dl class="dl-horizontal">';
    for (i = 0; i < order.length; i++) {
	k = order[i];
	v = dict[k];
	layout += ' <dt>'+k.toString()+':</dt>';
	layout += ' <dd>';
	layout += layout_value(v);
	layout += '</dd>';
    }
    layout += '</dl>';
    return layout;
}

function layout_value(v) {
    var i, layout = "";
    if (v == null) return "";
    if (Array.isArray(v)) {
	if (v.length === 0) return "";
	for (i = 0; i < v.length; i++) {
	    if (layout.length != 0)
		layout += "<br>";
	    layout += layout_value(v[i]);
	}
	    
    }
    if (typeof(v) === "object") {
	var keys = Object.keys(v);
	if (keys.length == 0) return "";
	return layout_dict(v, keys.sort());
    }
    return v.toString();
}
