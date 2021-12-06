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

import classifier
import cluster

classifier = classifier.Classifier()
clustering = cluster.Clustering()

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer

# Create HashingVectorizer instance
hv = HashingVectorizer(lowercase=False, alternate_sign = False)
cv = CountVectorizer(lowercase=False)

# Setting np seed to get reproducible models
np.random.seed(5)

	
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
		classifier.fit(hv, y)

		# Uncomment only 1 of these lines NOT BOTH. plot=True and False used as a flag for plotting clusters of each batch 
		# clustering.fit(hv, y, plot=False)
		clustering.fit(hv, y, plot=True)
		

if __name__ == "__main__":
	sc = SparkContext(appName="tweetStream")
	sc.setLogLevel("ERROR") # remove useless logs clogging the STDOUT
	ssc = StreamingContext(sc, batchDuration= 3)
	spark = SparkSession.builder.getOrCreate()
	classifier.initClassifiers()

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
	
	classifier.endClassifiers()
	clustering.endClustering()
