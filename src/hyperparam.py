from pyspark.sql.functions import col
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score


import pickle
import numpy as np

class Classifier:
    def __init__(self, filename = 'src/performance/training/supervised.csv'):
        self.batch = 0
        self.filename = filename

    def initClassifiers(self):
        self.GB_classifier1 = SGDClassifier(penalty = 'l1', n_jobs=-1)
        self.GB_classifier2 = SGDClassifier(penalty = 'l2', n_jobs=-1)
        self.NB_classifier1 = BernoulliNB(alpha = 1.0)
        # self.NB_classifier2 = BernoulliNB(alpha = 2.0)
        self.NB_classifier2 = BernoulliNB(alpha = 4.0)
        self.PA_classifier1 = PassiveAggressiveClassifier(C = 1.0, max_iter=1000, random_state=0)
        self.PA_classifier2 = PassiveAggressiveClassifier(C = 5.0, max_iter=1000, random_state=0)

        if self.batch == 0:
            with open(self.filename, 'w') as f:
                f.write('Batch No,SGD L1 Accuracy,SGD L2 Accuracy,NB alpha 1 Accuracy,NB alpha 4 Accuracy,PA C 1 Accuracy,PA C 5 Accuracy\n')
                f.close()
    
    def endClassifiers(self):
        pickle.dump(self.GB_classifier, open('GB.pkl', 'wb'))
        print("pickling GB successful")

        pickle.dump(self.NB_classifier, open('NB.pkl', 'wb'))
        print("pickling NB successful")

        pickle.dump(self.PA_classifier, open('PA.pkl', 'wb'))
        print("pickling PA successful")

    def fitSGD(self):
        self.GB_classifier1.partial_fit(self.X, self.Y, np.unique(self.Y))
        self.GB_classifier2.partial_fit(self.X, self.Y, np.unique(self.Y))
        score1 = self.GB_classifier1.score(self.X, self.Y)
        score2 = self.GB_classifier2.score(self.X, self.Y)
        # print(f"Batch {self.batch}, GB Accuracy:  {score}, F1 Score: {f1_SGD}, Precision: {prec_SGD}, Recall: {rec_SGD}")
        return score1, score2

    def fit_NB(self):
        self.NB_classifier1.partial_fit(self.X, self.Y, np.unique(self.Y))
        self.NB_classifier2.partial_fit(self.X, self.Y, np.unique(self.Y))
        score1 = self.NB_classifier1.score(self.X, self.Y)
        score2 = self.NB_classifier2.score(self.X, self.Y)
        # print(f"Batch {self.batch}, GB Accuracy:  {score}, F1 Score: {f1_SGD}, Precision: {prec_SGD}, Recall: {rec_SGD}")
        return score1, score2

    def fit_PA(self):
        self.PA_classifier1.partial_fit(self.X, self.Y, np.unique(self.Y))
        self.PA_classifier2.partial_fit(self.X, self.Y, np.unique(self.Y))
        score1 = self.PA_classifier1.score(self.X, self.Y)
        score2 = self.PA_classifier2.score(self.X, self.Y)
        # print(f"Batch {self.batch}, GB Accuracy:  {score}, F1 Score: {f1_SGD}, Precision: {prec_SGD}, Recall: {rec_SGD}")
        return score1, score2

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        sg1, sg2 = self.fitSGD()
        nb1, nb2 = self.fit_NB()
        pa1, pa2 = self.fit_PA()

        with open(self.filename, 'a') as f:
            f.write(f'{self.batch},{sg1},{sg2},{nb1},{nb2},{pa1},{pa2}\n')
            f.close()
        # Also write models
        self.endClassifiers()
        self.batch += 1