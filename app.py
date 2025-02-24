import eventlet
eventlet.monkey_patch()  # The eventlet.monkey_patch() function is used 
                         # to modify the standard Python libraries to enable asynchronous behavior

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS, cross_origin
from Neural_Network import train_neural_network
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Initialize SocketIO with eventlet
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
#cors_allowed_origins="*",
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
@cross_origin()#Fixed CORS issue for Socket.IO
def train():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden_size = 3
    output_size = 1
    epochs = 10000
    learning_rate = 0.1

    train_neural_network(X, Y, input_size, hidden_size, output_size, epochs, learning_rate, socketio)

    return jsonify({'image_path': '/static/loss_plot.png'})

if __name__ == '__main__':
    # Run the app with eventlet for asynchronous support
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)  # Use 0.0.0.0 for external access
