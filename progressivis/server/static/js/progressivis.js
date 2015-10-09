var ProgressiVis = {};

var socket = null;

function websocket_open() {
    socket = new WebSocket("ws://" + document.domain + ":5000/websocket/");

    socket.onopen = function() {
        socket.send("Joined");
    };

    socket.onmessage = function(message) {
        var txt = message.data;
        $("#error").append("<p>" + txt + "</p>");
    };
}

function websocket_submit(text) {
    socket.send(text);
}
