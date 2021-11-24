from sklearn.cluster import KMeans, MiniBatchKMeans

import numpy as np 

class Clustering:
    def __init__(self):
        self.km = MiniBatchKMeans(
            n_clusters=2,
            init="k-means++",
        )

    # def initCluster(self): 
    #     km = MiniBatchKMeans(
    #         n_clusters=2,
    #         init="k-means++",
    #     )

    def fit(self, X):
        self.km.partial_fit(X)
        print('Cluster centers:', self.km.cluster_centers_)

    def getClusterCenters(self):
        return self.km.cluster_centers_

    

    