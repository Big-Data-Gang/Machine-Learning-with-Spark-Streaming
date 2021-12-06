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
from sklearn.metrics import f1_score, precision_score, recall_score
from pipeline import PreProcess

import classifier
import cluster
import pickle

classifier = classifier.Classifier()
clustering = cluster.Clustering()

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer

# Create HashingVectorizer instance
hv = HashingVectorizer(lowercase=False, alternate_sign = False)
cv = CountVectorizer(lowercase=False)

# Load Models
nb = pickle.load('../models/2500/NB.pkl')
pa = pickle.load('../models/2500/PA.pkl')
sgd = pickle.load('../models/2500/GB.pkl')

# File to write
path = '../src/performance/testing/final-scores.csv'
# Write Headers
with open(path, 'w') as file:
    file.write('Batch,NB Score,NB F1,NB Precision,NB Recall,PA Score,PA F1,PA Precision,PA Recall,SGD Score,SGD F1,SGD Precision,SGD Recall\n')
    file.close()

# Batch
batch = 0

# Setting np seed to get reproducible models
np.random.seed(5)

def score(X, y):
    global batch
    global path
    
    # Naive Bayes
    nbScore = nb.score(X, y)
    y_pred_NB = nb.predict(X)
    nbF1 = f1_score(y, y_pred_NB)
    nbPrec = precision_score(y, y_pred_NB)
    nbRec = recall_score(y, y_pred_NB)

    # PA
    paScore = pa.score(X, y)
    y_pred_pa = pa.predict(X)
    paF1 = f1_score(y, y_pred_pa)
    paPrec = precision_score(y, y_pred_pa)
    paRec = recall_score(y, y_pred_pa)

    # SGD
    sgdScore = sgd.score(X, y)
    y_pred_sgd = sgd.predict(X)
    sgdF1 = f1_score(y, y_pred_sgd)
    sgdPrec = precision_score(y, y_pred_sgd)
    sgdRec = recall_score(y, y_pred_sgd)

    with open(path, 'a') as file:
        file.write(f'{batch},{nbScore},{nbF1},{nbPrec},{nbRec},{paScore},{paF1},{paPrec},{paRec},{sgdScore},{sgdF1},{sgdPrec},{sgdRec}\n')
        file.close()


def vectorize(df):
	# joined_df = df.withColumn('joined', array(col('finished')))
	joined_df = df.withColumn('joined', concat_ws(' ', col('finished')))
	docs = joined_df.select('joined').collect()
	corpus_batch = [doc['joined'] for doc in docs]

	hvect = hv.transform(corpus_batch)
	# cvect = cv.transform(corpus_batch)

	# return (hvect, cvect)
	return hvect

def process(rdd):
	# Array of elements of the dataset
	sent = rdd.collect()

	if len(sent) > 0:
		df = spark.createDataFrame(data=json.loads(sent[0]).values(), schema=['sentiment', 'tweet'])
		pipe = PreProcess(df)
		df = pipe()
		# vect = hv.fit_transform(df.select('finished').rdd.collect())
		# print(vect)
		# hv, cv = vectorize(df)
		hv = vectorize(df)

		y = np.array(df.select('sentiment').collect())
		y = np.reshape(y, (y.shape[0],))
        score(hv, y)
		

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
