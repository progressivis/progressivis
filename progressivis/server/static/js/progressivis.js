var ProgressiVis = {};

var socket = null,
    handshake = false,
    progressivis_run_number,
    progressivis_data,
    refresh, error;

function progressivis_websocket_open(msg, handler) {
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

function progressivis_update(data) {
    progressivis_data = data;
    progressivis_run_number = data['run_number'];
    $('#run_number').text(progressivis_run_number);
    if (data.is_running) {
        $('#start').addClass('disabled');
        $('#stop').removeClass('disabled');
    }
    else if (data.is_terminated) {
        $('#start').removeClass('disabled');
        $('#stop').removeClass('disabled');
    }
    else {
        $('#start').removeClass('disabled');
        $('#stop').addClass('disabled');
    }
}

function progressivis_websocket_submit(text) {
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

function progressivis_start(success, error) {
    $.post($SCRIPT_ROOT+'/progressivis/scheduler/start')
        .done(success)
        .fail(error);
};

function progressivis_stop(success, error) {
    $.post($SCRIPT_ROOT+'/progressivis/scheduler/stop')
        .done(success)
        .fail(error);
};

function progressivis_error(ev) {
  var contents = '<div class="alert alert-danger" role="alert">Error</div>';
  $('#error').html(contents);
}

function progressivis_socketmsg(message) {
    var txt = message.data,
        run_number;
    if (txt.startsWith("tick ")) {
        run_number = Number(txt.substr(5));
        //console.log('Reveived netsocket tick '+run_number);
        if (run_number > progressivis_run_number) {
            if (refresh == null) {
                console.log('ERROR: refresh is not defined');
            }
            else
                refresh();
        }
    }
    else 
        console.log('Received unexpected socket message: '+txt);
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
    $('#stop').click(function() { progressivis_stop(refresh, error); });
    progressivis_websocket_open(socket_name, progressivis_socketmsg);
}
