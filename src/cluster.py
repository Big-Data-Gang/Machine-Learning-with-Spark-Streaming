from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np 
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
import matplotlib.pyplot as plt


import pickle 

class Clustering:
    def __init__(self, filename = 'src/performance/training/unsupervised.csv'):
        self.batch = 0
        self.km = MiniBatchKMeans(
            n_clusters=2,
            init="k-means++",
        )
        self.tfidf = TfidfTransformer()
        self.ipca = IncrementalPCA(n_components=2)
        self.filename = filename
        if self.batch == 0:
            with open(self.filename, 'w') as f:
                f.write('Batch No,SSE,Silhouette Score,Accuracy\n')
                f.close()
        # if self.batch == 0:
        #     with open(self.filename, 'w') as f:
        #         f.write('Batch No,Centroid1,Centroid2\n')
        #         f.close()

    # def initCluster(self): 
    #     km = MiniBatchKMeans(
    #         n_clusters=2,
    #         init="k-means++",
    #     )
    def evaluate(self, actual, pred):
        newpred = np.where(pred == 1, 4, pred)
        corrects = np.where(newpred == actual, 1, 0)
        acc1 = np.count_nonzero(corrects)/len(corrects)
        return acc1

    def plot(self, X, pred):
        svd = TruncatedSVD(n_components = 2)
        reduced = svd.fit_transform(X)
        reduced_centroids = svd.fit_transform(self.getClusterCenters())

        if self.batch == 0:      
            plt.clf()

        print('No. of points in cluster 0', len(reduced[pred==0]))
        print('No. of points in cluster 1', len(reduced[pred==1]))

        plt.scatter(reduced[pred==0, 0], reduced[pred==0, 1], s=10, c='red', label ='Cluster 1')
        plt.scatter(reduced[pred==1, 0], reduced[pred==1, 1], s=10, c='blue', label ='Cluster 2')

        if self.batch == 303:
            plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], s=50, c='black', label = 'Centroids')
        else:
            plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], s=300, c='yellow', label = 'Centroids')

        plt.title(f'Batch {self.batch} clusters')
        fig = plt.gcf()
        fig.savefig(f'src/clustering_plots/training/batch{self.batch}.png', format ="png")

    def fit(self, X, y, plot=False):
        self.km.partial_fit(X)

        print('Cluster centers:', self.km.cluster_centers_)
        pred = self.km.predict(X)

        if plot:
            self.plot(X,pred)

        accuracy = self.evaluate(y, pred)
        sil_score =  silhouette_score(X, pred)
        print('Silhouette Score:', sil_score)
        print('Accuracy:', accuracy)

        with open(self.filename, 'a') as f:
            f.write(f'{self.batch},{self.km.inertia_},{sil_score},{accuracy}\n')
            f.close()

        self.batch += 1
        self.endClustering()

    def getClusterCenters(self):
        return self.km.cluster_centers_

    def endClustering(self):
        clusters = self.getClusterCenters()
        pickle.dump(clusters, open('clustercenters.pkl', 'wb'))
        pickle.dump(self.km, open('KMeans.pkl', 'wb'))
        print('Pickled KMeans')
        # with open('./src/performance/unsupervised-centroids.tsv', 'w') as f:
        #     for i, centroid in enumerate(clusters):
        #         f.write(f'Cluster{i}:\n')
        #         for j in centroid:
        #             f.write(f'{j}\t')
        #         f.write('\n\n\n\n')

    

    