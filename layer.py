import numpy as np


class DenseLayer:

    def __init__(self, n_inputs, n_neurons, L1w=0, L1b=0, L2w=0, L2b=0):
        """
        This initializes the weights to random (normally distributed) matrix of shape (n_inputs, n_neurons). Note
        that this is the weights transposed (saves us from having to do the later). The biases are initialized to an
        array of zeros of shape (1, n_neurons). The remaining parameter are the lambda regularization strength
        hyper-parameters with L1w being the lambda of the 'weights' for L1 regularization, etc.
        :param n_inputs: The number of inputs the layer receives.
        :param n_neurons: The number of neurons (nodes) in the layer.
        :param L1w: lambda of the weights for L1 regularization
        :param L1b: lambda of the biases for L1 regularization
        :param L2w: lambda of the weights for L2 regularization
        :param L2b: lambda of the biases for L1 regularization
        """
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.L1w = L1w
        self.L1b = L1b
        self.L2w = L2w
        self.L2b = L2b
        self.next = None
        self.prev = None

    def forward(self, inputs):
        """
        A forward pass of the data. This function first calculates the dot product of the weights and the inputs
        matricies. Then adds the bias. Finally it sets two new attributes: (1) the inputs being passed through the layer
        - so that this can be referenced later during backpropagation; (2) the output of the layer - so that this can be
        passed to the following activation function.
        :param inputs: the values being passed to the layer
        """
        output = np.dot(inputs, self.weights) + self.biases
        setattr(self, 'inputs', inputs)
        setattr(self, 'output', output)

    def backward(self, dvalues):
        """
        Calculates the derivative of the layer with respect to the weights, biases and inputs (dweights, dbiases,
        dinputs). Following on from this, if the lambda regularization constant is given, the derivative of the L1 and
        L2 regularization functions are calculated and added to their respective dweights and dbiases. Finally, the
        function sets these dervatives as attributes so that they can be accessed by other layers.
        :param dvalues: The derivatives received from the proceeding activation function
        """
        dweights = np.dot(self.inputs.T, dvalues)
        dbiases = np.sum(dvalues, axis=0, keepdims=True)
        dinputs = np.dot(dvalues, self.weights.T)

        if self.L1w > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            dweights += self.L1w * dL1

        if self.L2w > 0:
            dweights += 2 * self.L2w * self.weights

        if self.L1b > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            dbiases += self.L1b * dL1

        if self.L2b > 0:
            dbiases += 2 * self.L2b * self.biases

        setattr(self, 'dweights', dweights)
        setattr(self, 'dbiases', dbiases)
        setattr(self, 'dinputs', dinputs)
