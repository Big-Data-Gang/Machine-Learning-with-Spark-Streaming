#! /usr/bin/python3

TCP_IP = "localhost"
TCP_PORT = 6100

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from sparknlp.annotator import *
from sparknlp.base import *
import json
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pipeline import PreProcess

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveClassifier


import pickle
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Create HashingVectorizer instance
hv = HashingVectorizer(lowercase=False, alternate_sign = False)

# Setting np seed to get reproducible models
np.random.seed(5)

batch = 0

GB_classifier = None
NB_classifier = None
PA_classifier = None
def initClassifiers():
	global GB_classifier
	GB_classifier = SGDClassifier(n_jobs=-1) # defining globally
	NB_classifier = GaussianNB()
	PA_classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=0)

def endClassifiers():
	pickle.dump(GB_classifier, open('GB.pkl', 'wb'))
	print("pickling GB successful")

	pickle.dump(NB_classifier, open('NB.pkl', 'wb'))
	print("pickling NB successful")

	pickle.dump(PA_classifier, open('PA.pkl', 'wb'))
	print("pickling PA successful")
	

def fitSGD(X, y):
	GB_classifier.partial_fit(X, y, np.unique(y))
	print(f"Batch {batch}, GB Accuracy: ", GB_classifier.score(X, y))
	# print("fit one done")

def fit_NB(X, Y):
	NB_classifier.partial_fit(X, Y, np.unique(Y))
	print(f"Batch {batch}, NB Accuracy: ", NB_classifier.score(X, Y))

def fit_PA(X, Y):
	PA_classifier.partial_fit(X, Y, np.unique(Y))
	print(f"Batch {batch}, PA Accuracy: ", PA_classifier.score(X, Y))

def fit(X, Y):
	fit_NB(X, Y)
	fitSGD(X, Y)
	fit_PA(X, Y)
	global batch
	batch += 1
	
	
def vectorize(df):
	# joined_df = df.withColumn('joined', array(col('finished')))
	joined_df = df.withColumn('joined', concat_ws(' ', col('finished')))
	docs = joined_df.select('joined').collect()
	corpus_batch = [doc['joined'] for doc in docs]

	vect = hv.transform(corpus_batch).toarray()

	return vect

def process(rdd):
	# Array of elements of the dataset
	sent = rdd.collect()

	if len(sent) > 0:
		df = spark.createDataFrame(data=json.loads(sent[0]).values(), schema=['sentiment', 'tweet'])
		pipe = PreProcess(df)
		df = pipe()
		# vect = hv.fit_transform(df.select('finished').rdd.collect())
		# print(vect)
		vect = vectorize(df)
		y = np.array(df.select('sentiment').collect())
		y = np.reshape(y, (y.shape[0],))
		# print(vect.shape, y.shape)
		fit(vect, y)
		#df.show(truncate=False)
		

if __name__ == "__main__":
	sc = SparkContext(appName="tweetStream")
	sc.setLogLevel("ERROR") # remove useless logs clogging the STDOUT
	ssc = StreamingContext(sc, batchDuration= 3)
	spark = SparkSession.builder.getOrCreate()
	initClassifiers()

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
	
	endClassifiers()
	
