import numpy as np
from loss import CategoricalCrossEntropy
from activation import Softmax


class SoftmaxCCE:

    def __init__(self):
        self.loss = CategoricalCrossEntropy()
        self.activation = Softmax()
        self.prev = None
        self.next = None

    def forward(self, inputs, y_true, predict):
        """
        This function first calls the forward method of the Softmax activation function, then calls the forward method
        of the Categorical Cross entropy function. Function also saves the output of Softmax so that it can be used
        during back propagation.
        :param inputs: Output of the final layer
        :param y_true: The actual y values (determined by the labels)
        :param predict: A boolean that specifies if the model is training or predicting data
        :return: The model's loss or the model's prediction
        """
        self.activation.forward(inputs)
        setattr(self, 'output', self.activation.output)
        if predict:
            return self.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """
        The whole purpose of this class is to combine Softmax with CCE to gain performance increases during
        backpropagation. The combined derivative of Softmax and CCE can be reduced to the predicted values (y-pred)
        minus the target values (y_true). To implement the solution y-hat - y, instead of performing the subtraction of
        the full arrays, the function takes advantage of the fact that the y_true consists of one-hot encoded vectors,
        which means that, for each sample, there is only a singular value of 1 in these vectors and the remaining
        positions are filled with zeros. This means that we can use NumPy to index the prediction array with the sample
        number and its true value index, subtracting 1 from these values.
        :param dvalues: Since this is the last function we will be doing, dvalues = self.output
        :param y_true: The actual y values (determined by the label)
        """
        samples = len(dvalues)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        dinputs = dvalues.copy()
        dinputs[range(samples), y_true] -= 1
        dinputs /= samples
        setattr(self, 'dinputs', dinputs)


class StochasticGradientDecent:

    def __init__(self, learning_rate=1., decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_parameters(self):
        """
        A function that implements decay - is called before updating the parameters. This function updates the learning
        rate each step by the reciprocal of the step count fraction (learning rate decay). It takes the step and the
        decaying ratio and multiplies them. In effect this reduces the learning rate by an amount proportional to the
        epoch (the larger the epoch the lower the learning rate).
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_parameters(self, layer):
        """
        Function first checks if self.momentum is not zero. If the layer does not contain momentum arrays, create them.
        SGD utilizes momentum by setting a parameter between 0 and 1, representing the fraction of the previous
        parameter update to retain, and subtracting our actual gradient, multiplied by the learning rate, from it. If
        momentum is set to zero then just conduct vanilla gradient decent. Finally, the function updates the parameters
        of the layer.
        :param layer: a DenseLayer object
        """
        if self.momentum:
            # If layer does not contain momentum arrays, create them filled with zeros.
            if not hasattr(layer, 'weight_momentum'):
                setattr(layer, 'weight_momentum', np.zeros_like(layer.weights))
                setattr(layer, 'bias_momentum', np.zeros_like(layer.biases))
            weight_updates = (self.momentum * layer.weight_momentum) - (self.current_learning_rate * layer.dweights)
            bias_updates = (self.momentum * layer.bias_momentum) - (self.current_learning_rate * layer.dbiases)
            layer.weight_momentum = weight_updates
            layer.bias_momentum = bias_updates
        else:
            # Momentum is zero so just implement basic gradient decent
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_parameters(self):
        """
        Called after the parameters have been updated. Simply adds 1 to self.iterations
        """
        self.iterations += 1
