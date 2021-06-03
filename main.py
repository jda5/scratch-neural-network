from model import Model
import numpy as np
from mnist import MNIST

# Load the mnist data set

mndata = MNIST('mnist_data', return_type='numpy')
X, y = mndata.load_training()
X_test, y_test = mndata.load_testing()

# Scaling

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

# Create and train model

model = Model(learning_rate=0.004, decay=0, momentum=0.9)

model.add_layer(X.shape[1], 128, L2w=5e-4, L2b=5e-4)
model.add_layer(128, 128, L2w=5e-4, L2b=5e-4)
model.add_layer(128, 10)

model.link_nodes()

model.train(X, y, epochs=80, batch=128, print_frequency=1)

model.save('numeric_mnist.model')
print(model.evaluate(X_test, y_test))

