from pyspark.sql.functions import col
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
from sklearn import metrics


import pickle
import numpy as np

class Classifier:
    def __init__(self, filename = 'src/performance/supervised.csv'):
        self.batch = 0
        self.filename = filename

    def initClassifiers(self):
        self.GB_classifier = SGDClassifier(n_jobs=-1)
        self.NB_classifier = BernoulliNB()
        self.PA_classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=0)

        if self.batch == 0:
            with open(self.filename, 'w') as f:
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
        return score

    def fit_NB(self):
        self.NB_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        score = self.NB_classifier.score(self.X, self.Y)
        print(f"Batch {self.batch}, NB Accuracy: ", score)
        return score

    def fit_PA(self):
        self.PA_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        score = self.PA_classifier.score(self.X, self.Y)
        print(f"Batch {self.batch}, PA Accuracy: ", score)
        return score

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        scoreSGD = self.fitSGD()
        scoreNB = self.fit_NB()
        scorePA = self.fit_PA()

        with open(self.filename, 'a') as f:
            f.write(f'{self.batch},{scoreSGD},{scoreNB},{scorePA}\n')
            f.close()
        # Also write models
        self.endClassifiers()
        self.batch += 1