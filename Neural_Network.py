import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize neural network parameters
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.random.randn(hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.random.randn(output_size)
    return W1, b1, W2, b2
    
# Feedforward pass
def feedforward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1) 
    Z2 = np.dot(A1, W2) + b2 
    A2 = sigmoid(Z2)
    return A1, A2

# Backpropagation
def backpropagation(X, Y, A1, A2, W2):
    output_error = A2 - Y
    output_delta = output_error * sigmoid_derivative(A2)
    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(A1)
    return hidden_delta, output_delta

# Update weights and biases
def update_parameters(W1, b1, W2, b2, A1, X, hidden_delta, output_delta, learning_rate):
    W2 -= learning_rate * np.dot(A1.T, output_delta)
    b2 -= learning_rate * np.sum(output_delta, axis=0)
    W1 -= learning_rate * np.dot(X.T, hidden_delta)
    b1 -= learning_rate * np.sum(hidden_delta, axis=0)
    return W1, b1, W2, b2

# Loss function (Mean Squared Error)
def mean_squared_error(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred) ** 2)

# Train the neural network
def train_neural_network(X, Y, input_size, hidden_size, output_size, epochs, learning_rate, socketio):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    losses = []

    for epoch in range(epochs):
        A1, A2 = feedforward(X, W1, b1, W2, b2)
        loss = mean_squared_error(Y, A2)
        losses.append(loss)

        # Emit the loss to the frontend after each epoch
        socketio.emit('epoch_loss', {'epoch': epoch, 'loss': loss})

        hidden_delta, output_delta = backpropagation(X, Y, A1, A2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, A1, X, hidden_delta, output_delta, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    # Save the loss plot
    plt.plot(range(epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Time')
    
    image_path = os.path.join('static', 'loss_plot.png')
    plt.savefig(image_path)
    plt.close()

    return W1, b1, W2, b2
