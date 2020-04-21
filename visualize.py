from argparse import ArgumentParser
import os

import h5py
import json
import numpy as np
import tensorflow as tf

import common
import loss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import offsetbox
import skimage.transform as st

parser = ArgumentParser()

parser.add_argument('--embeddings',default='../triplet/market1501_test_embeddings.h5')
parser.add_argument('--dataset',default='data/market1501_test.csv')
parser.add_argument('--image_root',default='../Market-1501-v15.09.15')


args = parser.parse_args([])

def plot_embedding(X,y,images, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    ly = float(len(set(y)))
    for i in range(X.shape[0]):
        circle = plt.Circle((X[i,0],X[i,1]),0.001,color=plt.cm.Set1(y[i]/ly))
        ax.add_artist(circle)
        # plt.text(X[i, 0], X[i, 1], str(y[i]),
        #          color=plt.cm.Set1(y[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-4:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(st.resize(plt.imread(os.path.join(args.image_root,images[i])),(64,32)), cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

pids, fids = common.load_dataset(args.dataset, None)
with h5py.File(args.embeddings, 'r') as f:
        embs = np.array(f['emb'])
pids = np.array([int(i) for i in list(pids[:7000])])
# pca = PCA(n_components=50)
# pca_result = pca.fit_transform(embs)

tsne = TSNE(n_components=2, init='pca', random_state=0)
tsne_pca_results = tsne.fit_transform(embs[:7000])
plot_embedding(tsne_pca_results,pids,fids[:7000],title='TSNE plot')
plt.show()
# pca_result = np.concatenate((pca_result[:,0],pca_result[:,1],pca_result[:,2],pids))
# np.random.seed(42)
# rndperm = np.random.permutation(embs.shape[0])

# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=tsne_pca_results[rndperm,0], 
#     ys=tsne_pca_results[rndperm,1], 
#     zs=tsne_pca_results[rndperm,2], 
#     c=pids[rndperm], 
#     cmap='tab10'
# )
# ax.set_xlabel('pca-one')
# ax.set_ylabel('pca-two')
# ax.set_zlabel('pca-three')
# plt.show()
# plt.scatter(tsne_pca_results[rndperm,0],tsne_pca_results[rndperm,1],c=pids[rndperm],cmap='tab10')
# plt.show()
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=pca_result,
#     legend="full",
#     alpha=0.3
# )