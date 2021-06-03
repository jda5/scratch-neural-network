from optimizer import SoftmaxCCE, StochasticGradientDecent
from layer import DenseLayer
from activation import ReLU
from loss import Accuracy
from copy import deepcopy
import pickle
import numpy as np


class Model:

    def __init__(self, learning_rate=0.1, decay=0.01, momentum=0.9):
        self.layers = []
        self.activation_functions = []
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.optimizer = StochasticGradientDecent(learning_rate=learning_rate, decay=decay, momentum=momentum)
        self.accuracy = Accuracy()
        self.loss_activation = SoftmaxCCE()

    def add_layer(self, n_inputs, n_neurons, L1w=0., L1b=0., L2w=0., L2b=0.):
        self.layers.append(DenseLayer(n_inputs, n_neurons, L1w, L1b, L2w, L2b))

    def link_nodes(self):
        """
        Links nodes in a linked list data structure.
        """
        self.activation_functions = [ReLU() for _ in range(len(self.layers) - 1)]
        prev_activation = None
        for layer, activation in zip(self.layers, self.activation_functions):
            if prev_activation is not None:
                prev_activation.next = layer
            layer.prev = prev_activation
            layer.next = activation
            activation.prev = layer
            prev_activation = activation

        # Final layer
        self.layers[-1].prev = prev_activation
        prev_activation.next = self.layers[-1]
        self.layers[-1].next = self.loss_activation
        self.loss_activation.prev = self.layers[-1]

    def forward(self, layer, inputs, y, predict=False):
        if isinstance(layer, SoftmaxCCE):
            return layer.forward(inputs, y, predict)
        layer.forward(inputs)
        return self.forward(layer.next, layer.output, y, predict)

    def backward(self, layer, dvalues, y):
        if layer is None:
            return
        if isinstance(layer, SoftmaxCCE):
            layer.backward(dvalues, y)
        else:
            layer.backward(dvalues)
        return self.backward(layer.prev, layer.dinputs, y)

    def train(self, X, y, epochs=10, batch=None, print_frequency=1):
        if batch is not None:
            steps = X.shape[0] // batch
            if steps * batch < X.shape[0]:
                steps += 1
        else:
            steps = 1

        for epoch in range(epochs + 1):

            self.loss_activation.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(steps):
                if batch is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch:(step + 1) * batch]
                    batch_y = y[step * batch:(step + 1) * batch]

                # Forward Propagation
                self.forward(self.layers[0], batch_X, batch_y)

                # Determine accuracy
                self.accuracy.calculate(self.loss_activation.output, batch_y)

                # Back Propagation
                self.backward(self.loss_activation, self.loss_activation.output, batch_y)

                # Update parameters
                self.optimizer.pre_update_parameters()
                for layer in self.layers:
                    self.optimizer.update_parameters(layer)
                self.optimizer.post_update_parameters()

            # Get and print epoch loss and accuracy
            if epoch % print_frequency == 0:
                data_loss = self.loss_activation.loss.calculate_accumulated()
                regularization_loss = sum(map(self.loss_activation.loss.regularization_loss, self.layers))
                loss = data_loss + regularization_loss
                accuracy = self.accuracy.calculate_accumulated()

                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, data_loss: {data_loss:.3f}, '
                      f'reg_loss: {regularization_loss:.3f}, learning_rate: {self.optimizer.current_learning_rate: .5f}')

    def predict(self, X):
        """
        Conducts a single forward pass of the model with data given in X.
        :param X_test: Data to predict
        :return: The output of the Softmax function (a list of normalized probabilities)
        """
        # TODO: This function needs to check the format of the data
        output = self.forward(self.layers[0], X, None, predict=True)
        return output

    def save(self, path):
        """
        This function saves the model to the above specified path. First a copy is made of the model, then all internal
        counters and attributes of this copied model are reset or deleted (to allow for future training). Finally this
        model is saved using the Python pickle module https://docs.python.org/3/library/pickle.html
        :param path: The path and name of the file
        """
        model = deepcopy(self)
        model.loss_activation.loss.new_pass()
        model.accuracy.new_pass()
        # Clear all existing attributes
        for attribute in ('inputs', 'output', 'dinputs', 'dweights', 'dbiases'):
            for layer in model.layers:
                layer.__dict__.pop(attribute, None)
            for activation in model.activation_functions:
                activation.__dict__.pop(attribute, None)
            model.loss_activation.__dict__.pop(attribute, None)

        with open(path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load(path):
        """
        Loads a model at a specified path
        :param path: The path and filename of the pickle serialized model
        :return: The loaded model
        """
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model

    def evaluate(self, X_test, y_test):
        confidences = self.predict(X_test)
        predictions = np.argmax(confidences, axis=1)
        accuracy = np.mean(predictions == y_test)
        return accuracy
