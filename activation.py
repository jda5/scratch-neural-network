import numpy as np


class ReLU:

    def __init__(self):
        self.next = None
        self.prev = None

    def forward(self, inputs):
        """
        Implements Rectified Linear Unit (ReLU) function - all input values less than zero are replaced with zero.
        Finally it sets two new attributes: (1) the inputs being passed to the activation function - so that this can
        be referenced later during backpropagation; (2) the output of the layer - so that this can be passed on to the
        following layer.
        :param inputs: the values being passed to the activation function from the associated layer
        """
        output = np.maximum(0, inputs)
        setattr(self, 'output', output)
        setattr(self, 'inputs', inputs)

    def backward(self, dvalues):
        """
        The derivative of the ReLU function with respect to its inputs are zero if the input is less than zero. Since we
        are modifying the dvalues variable inplace, it's best to make a copy of them.
        :param dvalues: The derivatives received from the proceeding layer
        :return:
        """
        dinputs = dvalues.copy()
        dinputs[self.inputs < 0] = 0
        setattr(self, 'dinputs', dinputs)


class Softmax:

    def forward(self, inputs):
        """
        First the function exponentiates each value. However, it subtracts the largest of the inputs (row-wise) before
        doing the exponentiation to avoid the NaN trap. This creates un-normalized probabilities. Then we normalize
        these probabilities by dividing by the sum of the rows. Finally the output and input values are saves so that
        they can be referenced during backpropagation.
        :param inputs: the values being passed to the activation function from the associated layer
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        setattr(self, 'output', output)
        setattr(self, 'inputs', inputs)

# No need for a backwards function as this will be handled by the combined Softmax and Categorical Cross Entropy class
