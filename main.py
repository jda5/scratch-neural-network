from model import Model
import numpy as np
from mnist import MNIST
from dimensionality_reduction import PCA
from time import time
import matplotlib.pyplot as plt

# --- Load the mnist data set --- 

print("Loading data...")
mndata = MNIST('mnist-data', return_type='numpy')
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()
print('Running PCA...')

# --- Centering the data (scaling) ---

X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

# --- Dimensionality reduction ---

X = np.concatenate((X_train, X_test), axis=0)
preserve_var = 0.99
pca = PCA()
X_reduced = pca.reduce_dims(X, preserve_var=preserve_var)
print(f"--> Data reduced from {X.shape[1]} dimensions to {X_reduced.shape[1]} with {preserve_var*100}% of the variance preserved")
input()

# # --- Splitting Data ---

early_stopping = False

X_val = None
y_val = None
if early_stopping:
	X_train = X_reduced[:50000].astype(np.float32)
	X_test = X_reduced[60000:].astype(np.float32)
	X_val = X_reduced[50000:60000].astype(np.float32)
	y_val = y_train[50000:60000]
	y_train = y_train[:50000]
else:
	X_train = X_reduced[:60000].astype(np.float32)
	X_test = X_reduced[60000:].astype(np.float32)

# --- Create and train model ---

model = Model(learning_rate=0.007, decay=0.00001, momentum=0.9)

model.add_layer(X_train.shape[1], 128, L2w=5e-4, L2b=5e-4)
model.add_layer(128, 10)

model.link_nodes()

print("Training model...\n")
t0 = time()
model.train(X_train, y_train, epochs=15, batch=64, early_stopping=early_stopping, X_val=X_val, y_val=y_val)
t1 = time()

# model.save('numeric_mnist.model')
acc = model.evaluate(X_test, y_test)
print(f"\nModel trained for {'{:.2f}'.format(t1 - t0)} seconds, and scored {acc*100}% accuracy on the testing set.\n")
