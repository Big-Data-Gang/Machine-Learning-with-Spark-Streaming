#! /usr/bin/python3

TCP_IP = "localhost"
TCP_PORT = 6100

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql import Row
from sparknlp import DocumentAssembler
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline
import json
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import Word2Vec as wv
# IMPORTANT: NEVER CHANGE THE ABOVE TWO LINES

import pipeline

count = 0


def process(rdd):
	global count
	# Array of elements of the dataset
	sent = rdd.collect()

	if len(sent) > 0:
		df = spark.createDataFrame(data=json.loads(sent[0]).values(), schema=['sentiment', 'tweet'])
		df = pipeline.PreProcess(df)
		df.show(truncate=False)
		# count += df.count()
		# print(count)
		

if __name__ == "__main__":
	sc = SparkContext(appName="tweetStream")
	sc.setLogLevel("ERROR") # remove useless logs clogging the STDOUT
	ssc = StreamingContext(sc, batchDuration= 3)
	spark = SparkSession.builder.getOrCreate()

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
