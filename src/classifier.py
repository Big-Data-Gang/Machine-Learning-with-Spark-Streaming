from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveClassifier


import pickle
import numpy as np



class Classifier:
    def __init__(self):
        self.batch = 0

    def initClassifiers(self):
        self.GB_classifier = SGDClassifier(n_jobs=-1)
        self.NB_classifier = GaussianNB()
        self.PA_classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=0)
    
    def endClassifiers(self):
        pickle.dump(self.GB_classifier, open('GB.pkl', 'wb'))
        print("pickling GB successful")

        pickle.dump(self.NB_classifier, open('NB.pkl', 'wb'))
        print("pickling NB successful")

        pickle.dump(self.PA_classifier, open('PA.pkl', 'wb'))
        print("pickling PA successful")
        self.batch += 1

    def fitSGD(self):
        self.GB_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        print(f"Batch {self.batch}, GB Accuracy: ", self.GB_classifier.score(self.X, self.Y))
        # print("fit one done")

    def fit_NB(self):
        self.NB_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        print(f"Batch {self.batch}, NB Accuracy: ", self.NB_classifier.score(self.X, self.Y))

    def fit_PA(self):
        self.PA_classifier.partial_fit(self.X, self.Y, np.unique(self.Y))
        print(f"Batch {self.batch}, PA Accuracy: ", self.PA_classifier.score(self.X, self.Y))

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        self.fit_NB()
        self.fitSGD()
        self.fit_PA()
        self.batch += 1