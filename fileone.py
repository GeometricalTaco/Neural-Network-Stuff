from turtle import forward
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        ## Initialise weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        ## Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        ## Calculate output values from inputs
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        ## Get unnormalised probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))
        ## Normalise them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims = True)
        self.output = probabilities


class Loss:
    ## Calculates the data and regularisation losses given model output and ground truth values
    def calculate(self, output, y):
        ## Calculate sample losses
        sample_losses = self.forward(output, y)
        ##Calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CatagoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        ## number of samples in a batch
        samples = len(y_pred)

        ## Clip data to prevent division by 0
        ## Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        ## Probabilities for target values - only if categorical lables
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        ## Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        ## Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

## Create dataset
X,y = spiral_data(samples=100, classes=3)

## Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

## Create second dense layer with 3 input features (take output of previous layer) and 3 output values
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

## Create loss function
loss_function = Loss_CatagoricalCrossEntropy()

## Perform forward pass of training data through this layer
dense1.forward(X)
## Perform a forward pass through activation function
## It takes the output of first dense layer here
activation1.forward(dense1.output)

## Perform a forward pass through second dense layer, it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
## Perform a forward pass through activation function, takes the output of second dense layer here
activation2.forward(dense2.output)

## Output of first few samples
print(activation2.output[:5])

## Perform a forward pass through activation function, it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)

## Print loss value
print("Loss:", loss)