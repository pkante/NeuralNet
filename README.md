# NeuralNet
Neural network with just python and numpy, trained on MNIST dataset -> for learning purposes


## Dataset
I used the MNIST dataset, you can readjust for different datasets if needed

## Activation Functions
I used ReLU and softmax for activation functions, other options could be sigmoid, tanh, etc -> also depends on # of hidden layers
Why? -> This is because when you have weights and biases for a neuron thats essentially a linear combination, activation function allows you to take that linear combination "activate it" so it can then be used to predict nonlinear data

## Backprop
Purpose of backprop is go back and readjust/fine-tune the weights and biases of neural net to provide for better accuracy -> you can think of it like going backwards in the neural net and checking for which neurons have bad parameters

## Training
I trained the data by using a train/test split then evaluating the accuracy -> you can choose number of epochs and learning rate to be something else

## Improvement
You can improve this by adding more measures to prevent overfitting or trying different architectures/activation functions
