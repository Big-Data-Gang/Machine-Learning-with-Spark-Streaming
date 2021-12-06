from pyspark.sql.functions import col
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from sparknlp.annotator import *
from sparknlp.base import *
import json
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pipeline import PreProcess
#! /usr/bin/python3

TCP_IP = "localhost"
TCP_PORT = 6100


import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer

# Create HashingVectorizer instance
hv = HashingVectorizer(lowercase=False, alternate_sign = False)
cv = CountVectorizer(lowercase=False)

# Setting np seed to get reproducible models
np.random.seed(5)
import pickle
import numpy as np

class HyperParam:
    def __init__(self, filename = 'src/performance/hyperparam/supervised-run1.csv'):
        self.batch = 0
        self.filename = filename

    def initHyp(self):
        self.GB_classifier1 = SGDClassifier(penalty = 'l1', n_jobs=-1)
        self.GB_classifier2 = SGDClassifier(penalty = 'l2', n_jobs=-1)
        self.NB_classifier1 = BernoulliNB(alpha = 1.0)
        self.NB_classifier2 = BernoulliNB(alpha = 2.0)
        self.NB_classifier3 = BernoulliNB(alpha = 4.0)
        self.PA_classifier1 = PassiveAggressiveClassifier(C = 1.0, max_iter=1000, random_state=0)
        self.PA_classifier2 = PassiveAggressiveClassifier(C = 5.0, max_iter=1000, random_state=0)

        if self.batch == 0:
            with open(self.filename, 'w') as f:
                f.write('Batch No,SGD L1 Accuracy,SGD L2 Accuracy,NB alpha 1 Accuracy,NB alpha 2 Accuracy,NB alpha 4 Accuracy,PA C 1 Accuracy,PA C 5 Accuracy\n')
                f.close()
    
    def endHyp(self):
        pickle.dump(self.GB_classifier1, open('GB1.pkl', 'wb'))
        pickle.dump(self.GB_classifier2, open('GB2.pkl', 'wb'))
        print("pickling GB successful")

        pickle.dump(self.NB_classifier1, open('NB1.pkl', 'wb'))
        pickle.dump(self.NB_classifier2, open('NB2.pkl', 'wb'))
        pickle.dump(self.NB_classifier3, open('NB3.pkl', 'wb'))
        print("pickling NB successful")

        pickle.dump(self.PA_classifier1, open('PA1.pkl', 'wb'))
        pickle.dump(self.PA_classifier2, open('PA2.pkl', 'wb'))
        print("pickling PA successful")

    def fitSGD(self):
        self.GB_classifier1.partial_fit(self.X, self.Y, np.unique(self.Y))
        self.GB_classifier2.partial_fit(self.X, self.Y, np.unique(self.Y))
        score1 = self.GB_classifier1.score(self.X, self.Y)
        score2 = self.GB_classifier2.score(self.X, self.Y)
        print(f"Batch {self.batch}, GB1 Accuracy:  {score1}, GB2 Score: {score2}")
        return score1, score2

    def fit_NB(self):
        self.NB_classifier1.partial_fit(self.X, self.Y, np.unique(self.Y))
        self.NB_classifier2.partial_fit(self.X, self.Y, np.unique(self.Y))
        self.NB_classifier3.partial_fit(self.X, self.Y, np.unique(self.Y))
        score1 = self.NB_classifier1.score(self.X, self.Y)
        score2 = self.NB_classifier2.score(self.X, self.Y)
        score3 = self.NB_classifier3.score(self.X, self.Y)
        print(f"Batch {self.batch}, NB1 Accuracy:  {score1}, NB2 Score: {score2}, NB3 Score: {score3}")
        return score1, score2, score3

    def fit_PA(self):
        self.PA_classifier1.partial_fit(self.X, self.Y, np.unique(self.Y))
        self.PA_classifier2.partial_fit(self.X, self.Y, np.unique(self.Y))
        score1 = self.PA_classifier1.score(self.X, self.Y)
        score2 = self.PA_classifier2.score(self.X, self.Y)
        print(f"Batch {self.batch}, PA1 Accuracy:  {score1}, PA2 Score: {score2}")
        return score1, score2

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        sg1, sg2 = self.fitSGD()
        nb1, nb2, nb3 = self.fit_NB()
        pa1, pa2 = self.fit_PA()

        with open(self.filename, 'a') as f:
            f.write(f'{self.batch},{sg1},{sg2},{nb1},{nb2},{nb3},{pa1},{pa2}\n')
            f.close()
        # Also write models
        self.endHyp()
        self.batch += 1


hyp = HyperParam()
	
def vectorize(df):
	# joined_df = df.withColumn('joined', array(col('finished')))
	joined_df = df.withColumn('joined', concat_ws(' ', col('finished')))
	docs = joined_df.select('joined').collect()
	corpus_batch = [doc['joined'] for doc in docs]

	hvect = hv.transform(corpus_batch)
	return hvect

def process(rdd):
	# Array of elements of the dataset
	sent = rdd.collect()

	if len(sent) > 0:
		df = spark.createDataFrame(data=json.loads(sent[0]).values(), schema=['sentiment', 'tweet'])
		pipe = PreProcess(df)
		df = pipe()
		# vect = hv.fit_transform(df.select('finished').rdd.collect())
		hv = vectorize(df)

		y = np.array(df.select('sentiment').collect())
		y = np.reshape(y, (y.shape[0],))
		hyp.fit(hv, y)
		


if __name__ == "__main__":
	sc = SparkContext(appName="tweetStream")
	sc.setLogLevel("ERROR") # remove useless logs clogging the STDOUT
	ssc = StreamingContext(sc, batchDuration= 3)
	spark = SparkSession.builder.getOrCreate()
	hyp.initHyp()

	# Creates a DStream
	lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	# Transformation applied to each DStream iteration
	words = lines.flatMap(lambda line : line.split("\n"))
	
	words.foreachRDD(process)
	
	
	# Start the computation
	ssc.start()
	
	#wait till over
	ssc.awaitTermination()
	ssc.stop(stopGraceFully=True)
	
	hyp.endHyp()