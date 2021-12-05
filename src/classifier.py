from pyspark.sql.functions import col
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
from sklearn import metrics


import pickle
import numpy as np

# Columns list
cols = ['batch no', 'GB', 'NB', 'PA']

class Classifier:
    def __init__(self):
        self.batch = 0

    def initClassifiers(self):
        self.GB_classifier = SGDClassifier(n_jobs=-1)
        self.NB_classifier = BernoulliNB()
        self.PA_classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=0)

        # Create list to store data
        self.data = list()

        # Init dfs to save to csvs
        # self.csv = None

        if self.batch == 0:
            with open('src/performance/supervised.csv', 'w') as f:
                f.write('Batch No,SGD Accuracy,NB Accuracy,PA Accuracy\n')
                f.close()
    
    def endClassifiers(self):
        pickle.dump(self.GB_classifier, open('GB.pkl', 'wb'))
        print("pickling GB successful")

        pickle.dump(self.NB_classifier, open('NB.pkl', 'wb'))
        print("pickling NB successful")

        pickle.dump(self.PA_classifier, open('PA.pkl', 'wb'))
        print("pickling PA successful")

        # Save the csv files
        # batches = len(self.gb_csv)
        # self.gb_csv.to_csv(f'./src/performance/gb-{batches}.csv')
        # self.nb_csv.to_csv(f'./src/performance/nb-{batches}.csv')
        # self.pa_csv.to_csv(f'./src/performance/pa-{batches}.csv')
        # self.csv = pd.DataFrame(data=self.data, columns=cols)
        # self.csv.to_csv('./performance/supervised.csv', index=False)
        # self.batch += 1

    def fitSGD(self):
        self.GB_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        score = self.GB_classifier.score(self.X, self.Y)
        print(f"Batch {self.batch}, GB Accuracy: ", score)
        # d = {'Accuracy': self.GB_classifier.score(self.X, self.Y), 'Batch': self.batch}
        # df = pd.DataFrame(data=[[self.batch, self.GB_classifier.score(self.X, self.Y)]], columns=cols)
        # self.gb_csv = self.gb_csv.append(df)
        # print("fit one done")
        return score

    def fit_NB(self):
        self.NB_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        score = self.NB_classifier.score(self.X, self.Y)
        print(f"Batch {self.batch}, NB Accuracy: ", score)
        # df = pd.DataFrame(data=[[self.batch, self.NB_classifier.score(self.X, self.Y)]], columns=cols)
        # self.nb_csv = self.nb_csv.append(df)
        return score

    def fit_PA(self):
        self.PA_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        score = self.PA_classifier.score(self.X, self.Y)
        print(f"Batch {self.batch}, PA Accuracy: ", score)
        # df = pd.DataFrame(data=[[self.batch, self.PA_classifier.score(self.X, self.Y)]], columns=cols)
        # self.pa_csv = self.pa_csv.append(df)
        return score

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        scoreSGD = self.fitSGD()
        scoreNB = self.fit_NB()
        scorePA = self.fit_PA()
        
        print('Hi', scoreSGD, scoreNB, scorePA)

        with open('src/performance/supervised.csv', 'a') as f:
            print('In file')
            f.write(f'{self.batch},{scoreSGD},{scoreNB},{scorePA}\n')
            f.close()

        # self.data.append([self.batch, self.fitSGD(), self.fit_NB(), self.fit_PA()])
        self.batch += 1