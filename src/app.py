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

from sklearn.naive_bayes import GaussianNB
import pickle
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Create HashingVectorizer instance
hv = HashingVectorizer(lowercase=False)


clf_pf = None
def initClassifiers():
	global clf_pf
	clf_pf = GaussianNB() # defining globally

def endClassifiers():
	pickle.dump(clf_pf, open('NB.pkl', 'wb'))
	print("pickling sucessful")

def fitNB(df):
	#clf = GaussianNB()
	#clf.fit(X, Y)
	#GaussianNB()
	X = df.select("vector").rdd
	Y = df.select("sentiment").rdd
	clf_pf.partial_fit(X, Y, np.unique(Y))
	print("fit one done")
	
	
def vectorize(df):
	npArray = np.array(df.select('finished').collect())
	print(npArray[0])
	npArray = [i[0] for i in npArray]
	print(npArray[0])
	for i in npArray:
		for j in i:
			i = hv.fit_transform(j)
	print(npArray[0])
def process(rdd):
	# Array of elements of the dataset
	sent = rdd.collect()

	if len(sent) > 0:
		df = spark.createDataFrame(data=json.loads(sent[0]).values(), schema=['sentiment', 'tweet'])
		pipe = PreProcess(df)
		df = pipe()
		# vect = hv.fit_transform(df.select('finished').rdd.collect())
		# print(vect)
		vectorize(df)
		
		#fitNB(df)
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
	
