import random
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from collections import Counter
import torch.nn as nn

def visualize_tsne(x_train, labels, step):
	cls = len(Counter(labels))
	tsne = TSNE(n_components=2, verbose=1, random_state=123)
	z = tsne.fit_transform(x_train)
	df = pd.DataFrame()
	df["y"] = labels
	df["comp-1"] = z[:,0]
	df["comp-2"] = z[:,1]
	plt.figure()
	sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
			palette=sns.color_palette("hls", cls),
			data=df).set(title= f"T-SNE projection after {step} step")
	plt.savefig(f"tsne_{step}.png")
	
def visualize_tsne_labels(x_train, num_cls, step):
	num_ex = 20
	labels = []

	mul = int(num_ex / num_cls)
	rem = int(num_ex % num_cls)

	if rem == 0:
		for i in range(num_cls):
			labels = labels + [i] * mul
	else:
		extra_ex = np.random.choice(num_cls)
		for i in range(num_cls):
			if i == extra_ex:
				labels = labels + [i] * (mul+rem)
			else:
				labels = labels + [i] * mul
	        
	labels = np.array(labels*10)
	tsne = TSNE(n_components=2, verbose=1, random_state=123)
	z = tsne.fit_transform(x_train)
	df = pd.DataFrame()
	df["y"] = labels
	df["comp-1"] = z[:,0]
	df["comp-2"] = z[:,1]
	plt.figure()
	sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
			palette=sns.color_palette("hls", num_cls),
			data=df).set(title= f"T-SNE projection after {step} step")
	plt.savefig(f"tsne_{num_cls}_{step}.png")
	
def visualize_tsne_3d(x_train, labels, num_cls, step):
	# generate data
	# use tSNE to reduce dimension from 64 to 3
	tsne = make_pipeline(StandardScaler(), TSNE(n_components=3, init='pca', random_state=0))
	tsne.fit(x_train, labels)
	X_tsne_3d = tsne.fit_transform(x_train)
	#print('Dimensions after tSNE-3D:', X_tsne_3d.shape)

	# plot the points projected with PCA and tSNE
	fig = plt.figure()
	fig.suptitle(f"T-SNE projection after {step} step")
	ax = fig.add_subplot(111, projection='3d')
	ax.title.set_text('tSNE-3D')
	# get colormap from seaborn
	cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
	sc = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=labels, marker='o', cmap=cmap)
	ax.set_xlabel('x comp')
	ax.set_ylabel('y comp')
	ax.set_zlabel('z comp')
	#plt.show()

	# legend
	plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

	# save
	#plt.savefig("scatter_hue", bbox_inches='tight')
	plt.savefig(f"3d_tsne_{num_cls}_{step}", bbox_inches='tight')
	
	

	
	

     
    
	
	
