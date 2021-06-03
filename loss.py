import numpy as np


class Accuracy:

    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, output, y):
        """
        Function calculates the accuracy of a model's predictions. It first generates it's predictions based on the
        output with the highest probability. Then determines the average number of correct predictions.
        :param output: The output of the Softmax function (aka. self.loss_activation.output)
        :param y: The actual target values
        :return:
        """
        predictions = np.argmax(output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        self.accumulated_sum += np.sum(predictions == y)
        self.accumulated_count += len(predictions)
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self):  # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count  # Return the data and regularization losses
        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Loss:

    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, output, y_true):
        """
        A convenience function that calls the forward method of a loss function then determines the mean loss per sample
        Since training occurs in batches, the functions will account for this by tallying the accumulated sum and the
        accumulated count so that both batch-wise and epoch-wise statistics can be calculated.
        :param output: The output of the model (in this instance this is the output of the Softmax activation function
        of the final layer)
        :param y_true: The actual values (as determined by the labels)
        :return: The mean data loss per sample
        """
        sample_loss = self.forward(output, y_true)
        data_loss = np.mean(sample_loss)
        self.accumulated_sum += np.sum(sample_loss)
        self.accumulated_count += len(sample_loss)
        return data_loss

    def calculate_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    @staticmethod
    def regularization_loss(layer):
        """
        A function that calculates the loss due to regularization per layer.
        :param layer: A DenseLayer class
        :return: The loss due to regularization (as defined by the lambda values in the DenseLayer class)
        """
        loss = 0
        if layer.L1w > 0:
            loss += layer.L1w * np.sum(np.abs(layer.weights))
        if layer.L1b > 0:
            loss += layer.L1b * np.sum(np.abs(layer.biases))
        if layer.L2w > 0:
            loss += layer.L2w * np.sum(layer.weights * layer.weights)
        if layer.L2b > 0:
            loss += layer.L2b * np.sum(layer.biases * layer.biases)
        return loss


class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """
        Comapres a "ground truth" probability (y_true), to the predicted value (y_pred). First the predicted values are
        clipped, since log(0) is undefined and we want this clipping to be symmetrical (to avoid bias). Next, the
        function checks what form the target is in. They can be one-hot encoded, where all values, except for one, are
        zeros, and the correct label’s position is filled with 1. Or they can either be sparse, which means that the
        numbers they contain are the correct class number. The function then determines how confident it is with its
        prediction and finds its log.
        :param y_pred: The predictions made by the model
        :param y_true: The true values
        :return: Negative log-loss
        """
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            confidences = y_pred_clip[range(samples), y_true]

        # TODO: Delete this later
        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clip * y_true, axis=1)

        loss = -np.log(confidences)

        return loss

# No need for a backwards function as this will be handled by the combined Softmax and Categorical Cross Entropy class
