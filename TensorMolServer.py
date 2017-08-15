'''
sudo pip install redis flask gevent flask-socketio
also execute redis-server
'''
#from TensorMol import *
from __future__ import absolute_import
from gevent import monkey
monkey.patch_all()
import redis
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
db = redis.StrictRedis('localhost',6379,0)
socketio = SocketIO(app)

@app.route('/')
def main():
	c = db.incr('counter')
	return render_template('main.html', counter = c)

@socketio.on('connect', namespace='/tms')
def ws_conn():
	c = db.incr('user_count')
	socketio.emit('msg',{'count':c}, namespace = '/tms')

@socketio.on('disconnect', namespace='/tms')
def ws_disconn():
        c = db.decr('user_count')
        socketio.emit('msg',{'count':c}, namespace = '/tms')

if __name__=='__main__':
	#app.run(debug=True)
	socketio.run(app)
