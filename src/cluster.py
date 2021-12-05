from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np 
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA

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
                f.write('Batch No,SSE,Silhouette Score\n')
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

    def fit(self, X):
        vect = self.tfidf.fit_transform(X)
        # self.ipca.partial_fit(X)
        # vect = self.ipca.transform(X)
        self.km.partial_fit(vect)

        print('Cluster centers:', self.km.cluster_centers_)
        # with open(self.filename, 'a') as f:
        #         f.write(f'{self.batch},{self.km.cluster_centers_[0]},{self.km.cluster_centers_[1]}\n')
        #         f.close()
        pred = self.km.predict(vect)
        print('Silhouette Score:', silhouette_score(vect, pred))

        with open(self.filename, 'a') as f:
            f.write(f'{self.batch},{self.km.inertia_},{silhouette_score(vect, pred)}\n')
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

    

    