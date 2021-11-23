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
import pickle
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Create HashingVectorizer instance
hv = HashingVectorizer(lowercase=False)

# Setting np seed to get reproducible models
np.random.seed(5)

count = 0

clf_pf = None
def initClassifiers():
	global clf_pf
	clf_pf = SGDClassifier(n_jobs=-1) # defining globally

def endClassifiers():
	pickle.dump(clf_pf, open('NB.pkl', 'wb'))
	print("pickling sucessful")

def fitNB(X, y):
	global count
	#clf = GaussianNB()
	#clf.fit(X, Y)
	#GaussianNB()
	clf_pf.partial_fit(X, y, np.unique(y))
	print(f"Batch {count} Accuracy: ", clf_pf.score(X, y))
	# print("fit one done")
	count += 1
	
	
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
		fitNB(vect, y)
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
	
