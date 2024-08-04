import numpy as np

# Load MNIST dataset -> for user to download
train_images = np.load('train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('train_labels.npy')  # Shape: (60000,)
test_images = np.load('test_images.npy')    # Shape: (10000, 784)
test_labels = np.load('test_labels.npy')    # Shape: (10000,)

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

#relu for activation function
def relu(x):
    return np.maximum(0, x)

#used for backprop
def relu_derivative(x):
    return (x > 0).astype(float)

#2nd activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#initialize weights and biases for neurons
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

#forward pass w/ the activation functions
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#loss function -> cross-entropy loss
def compute_loss(Y, A2):
    m = Y.shape[0]
    log_probs = -np.log(A2[range(m), Y])
    loss = np.sum(log_probs) / m
    return loss

#backprop -> readjusts weights and biases for neuron to increase accuracy
def backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = X.shape[0]
    
    dZ2 = A2
    dZ2[range(m), Y] -= 1
    dZ2 /= m
    
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2

#adjust params to increase accuracy
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def train(X_train, Y_train, input_size, hidden_size, output_size, epochs, learning_rate):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)
        loss = compute_loss(Y_train, A2)
        
        dW1, db1, dW2, db2 = backpropagation(X_train, Y_train, Z1, A1, Z2, A2, W1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

def accuracy(predictions, labels):
    return np.mean(predictions == labels)

# Hyperparameters
input_size = 784  # 28x28 images flattened -> applies for MNIST dataset
hidden_size = 128
output_size = 10  # 10 classes for digits 0-9 -> applies for MNIST dataset
epochs = 1000
learning_rate = 0.01

# Training model
W1, b1, W2, b2 = train(train_images, train_labels, input_size, hidden_size, output_size, epochs, learning_rate)

# Evaluating model
train_preds = predict(train_images, W1, b1, W2, b2)
test_preds = predict(test_images, W1, b1, W2, b2)

print(f'Training Accuracy: {accuracy(train_preds, train_labels) * 100:.2f}%')
print(f'Test Accuracy: {accuracy(test_preds, test_labels) * 100:.2f}%')
