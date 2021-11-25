from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
from sklearn import metrics


import pickle
import numpy as np

# Columns list
cols = ['batch no', 'accuracy']

class Classifier:
    def __init__(self):
        self.batch = 0

    def initClassifiers(self):
        self.GB_classifier = SGDClassifier(n_jobs=-1)
        self.NB_classifier = GaussianNB()
        self.PA_classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=0)

        # Init dfs to save to csvs
        self.gb_csv = pd.DataFrame(columns=cols)
        self.nb_csv = pd.DataFrame(columns=cols)
        self.pa_csv = pd.DataFrame(columns=cols)
    
    def endClassifiers(self):
        pickle.dump(self.GB_classifier, open('GB.pkl', 'wb'))
        print("pickling GB successful")

        pickle.dump(self.NB_classifier, open('NB.pkl', 'wb'))
        print("pickling NB successful")

        pickle.dump(self.PA_classifier, open('PA.pkl', 'wb'))
        print("pickling PA successful")

        # Save the csv files
        self.gb_csv.to_csv('./src/performance/gb.csv')
        self.nb_csv.to_csv('./src/performance/nb.csv')
        self.pa_csv.to_csv('.src//performance/pa.csv')
        self.batch += 1

    def fitSGD(self):
        self.GB_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        print(f"Batch {self.batch}, GB Accuracy: ", self.GB_classifier.score(self.X, self.Y))
        # d = {'Accuracy': self.GB_classifier.score(self.X, self.Y), 'Batch': self.batch}
        df = pd.DataFrame(data=[[self.batch, self.GB_classifier.score(self.X, self.Y)]], columns=cols)
        self.gb_csv = self.gb_csv.append(df)
        # print("fit one done")

    def fit_NB(self):
        self.NB_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        print(f"Batch {self.batch}, NB Accuracy: ", self.NB_classifier.score(self.X, self.Y))
        df = pd.DataFrame(data=[[self.batch, self.NB_classifier.score(self.X, self.Y)]], columns=cols)
        self.nb_csv = self.nb_csv.append(df)

    def fit_PA(self):
        self.PA_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        print(f"Batch {self.batch}, PA Accuracy: ", self.PA_classifier.score(self.X, self.Y))
        df = pd.DataFrame(data=[[self.batch, self.PA_classifier.score(self.X, self.Y)]], columns=cols)
        self.pa_csv = self.pa_csv.append(df)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        #self.fit_NB()
        self.fitSGD()
        self.fit_PA()
        self.batch += 1