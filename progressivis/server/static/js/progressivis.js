var ProgressiVis = {};

var socket = null,
    handshake = false,
    protocol = null,
    progressivis_run_number,
    progressivis_data,
    refresh, error;

function progressivis_websocket_open(msg, handler) {
    socket = new WebSocket("ws://" + document.domain + ":5000/websocket/", "new");
    //socket = new WebSocket("ws://" + document.domain + ":5000/websocket/");

    socket.onopen = function() {
        protocol = socket.protocol;
        if (protocol)
            console.log('Socket open with protocol '+protocol);
        if (protocol == "new")
            socket.send(JSON.stringify({"type": "ping", "path": msg}));
        else
            socket.send("ping "+msg);
	handshake = false;
    };

    socket.onclose = function() {
        handshake = false;
    };

    socket.onmessage = function(message) {
        if (protocol == "new") {
            var msg = JSON.parse(message.data);
            if (msg.type == "pong" && ! handshake) {
	        handshake = true;
	        console.log('Handshake received');
	        return;
            }
            if (handler) handler(msg);
        }
        // var txt = message.data;
	// if (txt == 'pong' && !handshake) {
	//     handshake = true;
	//     console.log('Handshake received');
	//     return;
	// }
	// if (handler) handler(message);
    };
}

function progressivis_update(data) {
    progressivis_data = data;
    progressivis_run_number = data['run_number'];
    $('#run_number').text(progressivis_run_number);
}

function progressivis_websocket_submit(text) {
    if (handshake == true)
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

function escapeHTML(s) {
    return $('<div>').text(s).html();
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
        return layout;
    }
    if (typeof(v) === "string" && v.startsWith('<div')) {
        return v;
    }
    if (typeof(v) === "object") {
	var keys = Object.keys(v);
	if (keys.length == 0) return "";
	return layout_dict(v, keys.sort());
    }
    return escapeHTML(v.toString()); // escape
}

function progressivis_post(url, success, error) {
    var xdr = $.post(url);
    if (success)
        xdr.done(success);
    if (error)
        xdr.fail(error);
    return xdr;
}

function progressivis_start(success, error) {
    return progressivis_post($SCRIPT_ROOT+'/progressivis/scheduler/start',
                             success, error);
}

function progressivis_stop(success, error) {
    return progressivis_post($SCRIPT_ROOT+'/progressivis/scheduler/stop',
                             success, error);
}

function progressivis_step(success, error) {
    return progressivis_post($SCRIPT_ROOT+'/progressivis/scheduler/step',
                             success, error);
}

function progressivis_error(ev, msg) {
    var contents = '<div class="alert alert-danger alert-dismissible" role="alert">Error: ';
    if (msg)
        contents += msg;
    contents += '</div>';
  $('#error').html(contents);
}

function progressivis_socketmsg(json) {
    var run_number = json.run_number;

    if (run_number > progressivis_run_number) {
        progressivis_run_number = run_number;
        if (refresh == null) {
            console.log('ERROR: refresh is not defined');
        }
        else
            refresh(json);
    }

    // if (txt.startsWith("tick ")) {
    //     run_number = Number(txt.substr(5));
    //     //console.log('Reveived netsocket tick '+run_number);
    //     if (run_number > progressivis_run_number) {
    //         progressivis_run_number = run_number;
    //         if (refresh == null) {
    //             console.log('ERROR: refresh is not defined');
    //         }
    //         else
    //             refresh();
    //     }
    // }
    // else 
    //     console.log('Received unexpected socket message: '+txt);
}

function progressivis_ready(socket_name) {
    if (error === null) 
        error = progressivis_error;
    if (refresh === null) {
        console.log('ERROR: refresh is not defined');
    }
    else
        refresh();
    $('#start').click(function() { progressivis_start(refresh, error); });
    $('#stop' ).click(function() { progressivis_stop (refresh, error); });
    $('#step' ).click(function() { progressivis_step (refresh, error); });
    progressivis_websocket_open(socket_name, progressivis_socketmsg);
}

window.addEventListener('visibilitychange', function() {if(!document.hidden){refresh();}});
