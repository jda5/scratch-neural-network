import numpy as np

class PCA:

	def reduce_dims(self, X, preserve_var=0.99):
		"""
		preserve_var: takes an float value between 0 and 1. Indicates the ratio of variance to preserve. 
		"""
		# Get eigenvalues and eigenvectors by calculating the covariance matrix
		# TODO: Get a performance increase by using Singular Value Decomposition
		cov_mat = np.cov(X.T)
		eig_vals, eig_vecs = np.linalg.eig(cov_mat)

		# Get explained variance and determined the number of dimensions to reduce the data to
		var_exp = sorted(eig_vals / np.sum(eig_vals), reverse=True)
		cum_var_exp = np.cumsum(var_exp)
		d = np.argmax(cum_var_exp >= preserve_var) + 1

		# Create a projection matrix
		w = eig_vecs[:,:d]
		
		# Return reduced dimensionality dataset
		return X.dot(w)
