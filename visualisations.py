import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter:

	def __init__(self):
		self.model_accuracy = []
		self.val_accuracy = []

	def save_figure(self, fig, file_name, file_type='png', resolution=1000):
		file_path = os.path.join('.', file_name + "." + file_type)
		fig.savefig(file_path, format=file_type, dpi=resolution)

	def draw_figure(self):
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(111)

		ax.set_ylabel('Accuracy')
		ax.set_xlabel('Epochs')

		x = np.arange(len(self.model_accuracy))
		ax.plot(x, self.val_accuracy, 'orange', label='Validation', zorder=3)
		ax.plot(x, self.model_accuracy, 'blue', label='Model', zorder=3)
		ax.legend(loc="lower right")
		
		ax.grid(b=True, zorder=0)

		self.save_figure(fig, 'figure')

# Leftover 2-D MNIST Projection

# import matplotlib.patches as mpatches

# colors = {0: '#fff100', 1: '#ff8c00', 2: '#e81123', 3: '#ec008c', 4: '#68217a', 5: '#00188f', 6: '#00bcf2', 7:'#00b294', 
# 	8:'#009e49', 9: '#bad80a'}

# handles = []
# for key, value in colors.items():
# 	patch = mpatches.Patch(color=value, label=str(key))
# 	handles.append(patch)

# labels = np.concatenate((y_train, y_test), axis=0)

# def color_map(x):
# 	res = []
# 	for i in x:
# 		res.append(colors[i])

# 	return res

# x = X_reduced.T[0].astype(np.float32)
# y = X_reduced.T[1].astype(np.float32)

# colors = color_map(labels)

# fig = plt.figure(figsize=(12,10))
# ax = fig.add_subplot(111)
# ax.set_ylabel('Second Principal Component')
# ax.set_xlabel('First Principal Component')
# ax.set_title('PCA 2-D Projection')

# ax.scatter(x, y, s=2.5, c=colors, alpha=0.6)
# ax.legend(handles=handles, bbox_to_anchor=(1.01, 1.0), loc='upper left')
# fig.savefig('mnist.png', format='png', dpi=1000)
