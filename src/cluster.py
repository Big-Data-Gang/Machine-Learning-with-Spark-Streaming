from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np 
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
import matplotlib.pyplot as plt

# cols = ['batch no', 'accuracy']

import pickle 

class Clustering:
    def __init__(self, filename = 'src/performance/unsupervised.csv'):
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

        # self.batch = 0
        # self.csv = None
        # self.data = list()

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

        # print('Hello', pred)
        # print(pred==0)
        # print(pred==1)
        # print('Hiiii', reduced, '\n', reduced[pred==0])

        if self.batch != 0:      
            plt.clf()

        print('Hi', len(reduced[pred==0]), reduced[pred==0])
        print('Hi2', len(reduced[pred==1]), reduced[pred==1])

        plt.scatter(reduced[pred==0, 0], reduced[pred==0, 1], s=50, c='red', label ='Cluster 1')
        plt.scatter(reduced[pred==1, 0], reduced[pred==1, 1], s=50, c='blue', label ='Cluster 2')
        plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], s=200, c='black', label = 'Centroids')
        plt.title(f'Batch {self.batch} clusters')
        fig = plt.gcf()
        fig.savefig(f'src/clustering_plots/training/batch{self.batch}.png', format ="png")
        # print('Hello')
        # for i in zip(pred, reduced):
        #     print(i)

    def fit(self, X, y, plot=False):
        vect = X
        # vect = self.tfidf.fit_transform(X)
        # self.ipca.partial_fit(X)
        # vect = self.ipca.transform(X)
        self.km.partial_fit(vect)

        print('Cluster centers:', self.km.cluster_centers_)
        # with open(self.filename, 'a') as f:
        #         f.write(f'{self.batch},{self.km.cluster_centers_[0]},{self.km.cluster_centers_[1]}\n')
        #         f.close()
        pred = self.km.predict(vect)

        if plot:
            self.plot(vect,pred)

        # print('Helloo', len(pred), pred)
        accuracy = self.evaluate(y, pred)
        sil_score =  silhouette_score(vect, pred)
        print('Silhouette Score:', sil_score)
        print('Accuracy:', accuracy)

        with open(self.filename, 'a') as f:
            f.write(f'{self.batch},{self.km.inertia_},{sil_score},{accuracy}\n')
            f.close()

        self.batch += 1
        self.endClustering()
        # self.data.append(self.km.cluster_centers_)

    def getClusterCenters(self):
        return self.km.cluster_centers_

    def endClustering(self):
        clusters = self.getClusterCenters()
        pickle.dump(clusters, open('clustercenters.pkl', 'wb'))
        pickle.dump(self.km, open('KMeans.pkl', 'wb'))
        print('Pickled KMeans')
        with open('./src/performance/unsupervised-centroids.tsv', 'w') as f:
            for i, centroid in enumerate(clusters):
                f.write(f'Cluster{i}:\n')
                for j in centroid:
                    f.write(f'{j}\t')
                f.write('\n\n\n\n')

    

    