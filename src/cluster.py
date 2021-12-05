from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import numpy as np 

cols = ['batch no', 'accuracy']


class Clustering:
    def __init__(self):
        self.km = MiniBatchKMeans(
            n_clusters=2,
            init="k-means++",
        )
        # self.batch = 0
        # self.csv = None
        # self.data = list()

    # def initCluster(self): 
    #     km = MiniBatchKMeans(
    #         n_clusters=2,
    #         init="k-means++",
    #     )

    def fit(self, X):
        self.km.partial_fit(X)
        print('Cluster centers:', self.km.cluster_centers_)
        # self.data.append(self.km.cluster_centers_)

    def getClusterCenters(self):
        return self.km.cluster_centers_

    

    