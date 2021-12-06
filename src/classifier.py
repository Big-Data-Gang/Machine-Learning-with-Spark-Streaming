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
        self.GB_classifier = SGDClassifier(n_jobs=-1)
        self.NB_classifier = BernoulliNB()
        self.PA_classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=0)

        if self.batch == 0:
            with open(self.filename, 'w') as f:
                f.write('Batch No,SGD Accuracy,SGD F1,SGD Precision,SGD Recall,NB Accuracy,NB F1,NB Precision,NB Recall,PA Accuracy,PA F1,PA Precision,PA Recall,\n')
                f.close()
    
    def endClassifiers(self):
        pickle.dump(self.GB_classifier, open('GB.pkl', 'wb'))
        print("pickling GB successful")

        pickle.dump(self.NB_classifier, open('NB.pkl', 'wb'))
        print("pickling NB successful")

        pickle.dump(self.PA_classifier, open('PA.pkl', 'wb'))
        print("pickling PA successful")

    def fitSGD(self):
        self.GB_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        score = self.GB_classifier.score(self.X, self.Y)
        y_pred = self.GB_classifier.predict(self.X)
        f1_SGD = f1_score(self.Y, y_pred, pos_label=4)
        prec_SGD = precision_score(self.Y, y_pred, pos_label=4)
        rec_SGD = recall_score(self.Y, y_pred, pos_label=4)
        print(f"Batch {self.batch}, GB Accuracy:  {score}, F1 Score: {f1_SGD}, Precision: {prec_SGD}, Recall: {rec_SGD}")
        return score, f1_SGD, prec_SGD, rec_SGD

    def fit_NB(self):
        self.NB_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        score = self.NB_classifier.score(self.X, self.Y)
        print(f"Batch {self.batch}, NB Accuracy: ", score)
        y_pred = self.NB_classifier.predict(self.X)
        f1_NB = f1_score(self.Y, y_pred, pos_label=4)
        prec_NB = precision_score(self.Y, y_pred, pos_label=4)
        rec_NB = recall_score(self.Y, y_pred, pos_label=4)
        print(f"Batch {self.batch}, GB Accuracy:  {score}, F1 Score: {f1_NB}, Precision: {prec_NB}, Recall: {rec_NB}")
        return score, f1_NB, prec_NB, rec_NB

    def fit_PA(self):
        self.PA_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        score = self.PA_classifier.score(self.X, self.Y)
        print(f"Batch {self.batch}, PA Accuracy: ", score)
        y_pred = self.PA_classifier.predict(self.X)
        f1_PA = f1_score(self.Y, y_pred, pos_label=4)
        prec_PA = precision_score(self.Y, y_pred, pos_label=4)
        rec_PA = recall_score(self.Y, y_pred, pos_label=4)
        print(f"Batch {self.batch}, GB Accuracy:  {score}, F1 Score: {f1_PA}, Precision: {prec_PA}, Recall: {rec_PA}")
        return score, f1_PA, prec_PA, rec_PA

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        scoreSGD, f1SGD, precSGD, recSGD = self.fitSGD()
        scoreNB, f1NB, precNB, recNB = self.fit_NB()
        scorePA, f1PA, precPA, recPA = self.fit_PA()

        with open(self.filename, 'a') as f:
            f.write(f'{self.batch},{scoreSGD},{f1SGD},{precSGD},{recSGD},{scoreNB},{f1NB},{precNB},{recNB},{scorePA},{f1PA},{precPA},{recPA}\n')
            f.close()
        # Also write models
        self.endClassifiers()
        self.batch += 1