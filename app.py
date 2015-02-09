from gevent import monkey
monkey.patch_all()

import time
from threading import Thread
from flask import Flask, render_template, session, request, Response
from flask.ext.socketio import SocketIO, emit, join_room, leave_room, \
    close_room, disconnect

from progressivemds.rand import random_mds

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
thread = None

def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        time.sleep(10)
        count += 1
        socketio.emit('my response',
                      {'data': 'Server generated event', 'count': count},
                      namespace='/mds')

@app.route("/")
def index():
    global thread
    if thread is None:
        thread = Thread(target=background_thread)
        thread.start()
    return render_template('index.html')

@app.route("/data/mds/<int:seed>")
def data(seed):
    df = random_mds(100, seed)
    csv = df.to_csv(sep='\t',
                    header=True,
                    index=True,
                    index_label="id",
                    encoding="utf-8")
    #print csv
    return Response(csv, "Content-type: text/plain")


@socketio.on('my event', namespace='/mds')
def test_message(message):
    emit('my response', {'data': message['data']})

@socketio.on('my broadcast event', namespace='/mds')
def test_message(message):
    emit('my response', {'data': message['data']}, broadcast=True)

@socketio.on('connect', namespace='/mds')
def test_connect():
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/mds')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    app.debug = True
    socketio.run(app)
