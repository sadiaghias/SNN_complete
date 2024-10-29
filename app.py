from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from Neural_Network import train_neural_network  # Import the training function
import numpy as np
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="http://127.0.0.1:5000")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    # Input and output data for XOR problem
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
    socketio.run(app, debug=True)  # Run the app with SocketIO

