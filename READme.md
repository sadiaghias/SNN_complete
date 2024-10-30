## Simple Neural Network Training with Flask and Socket.IO

This project demonstrates a simple neural network built from scratch using Python, trained to solve the XOR problem. The model is built using NumPy and provides real-time feedback on training progress by emitting the loss value after each epoch to a frontend client via Flask and Socket.IO.

## Project Structure

- **`app.py`**: The main Flask application file which sets up the server and endpoints.
- **`Neural_Network.py`**: Contains the neural network's functions, including initialization, feedforward, backpropagation, and training functions.
- **`templates/index.html`**: The frontend HTML page that displays real-time loss values and the final loss plot.
- **`static/loss_plot.png`**: Location where the loss plot is saved and displayed in the frontend after training is complete.

## Features

- **Real-time Loss Updates**: The backend emits the loss value after each epoch to the frontend, which is displayed in real time.
- **Loss Plot Display**: After training is completed, a plot of loss over time is generated and displayed on the frontend.
- **Basic Neural Network**: Built from scratch without external deep learning libraries, using only NumPy.

## Setup

### Prerequisites

- Python 3.x
- Flask
- Flask-SocketIO
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sadiaghias/SNN_complete.git
   cd your-repo-name


2. Install the required packages:
pip install Flask Flask-SocketIO numpy matplotlib


## Start the Flask app:

python app.py
Open a browser and go to http://127.0.0.1:5000.
Click "Start Training" to initiate the neural network training process.

## How it Works:
The neural network is initialized with weights and biases and trained over a specified number of epochs.
After each epoch, the backend sends the epoch number and current loss to the frontend using Socket.IO.
Once training completes, a loss plot is saved to static/loss_plot.png and displayed on the frontend.

## Example Output:
Epoch Loss: Displayed in real time on the frontend.
Loss Plot: A plot showing the trend of loss over epochs is displayed after training.

## Troubleshooting
1.Connection issues: 
Ensure that Flask-SocketIO is properly installed and configured with cors_allowed_origins.
2.Missing Image: 
If the loss plot does not display, make sure static/loss_plot.png was saved correctly.