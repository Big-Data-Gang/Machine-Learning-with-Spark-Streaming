#! /usr/bin/python3

TCP_IP = "localhost"
TCP_PORT = 6100

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql import Row
import json
from pyspark.sql.types import StructType,StructField, StringType

# Dataframe Schema
schema = StructType([
  StructField('sentiment', StringType(), True),
  StructField('text', StringType(), True),
  ])



def predict(tweet_text):
	
	sent = tweet_text.collect()
	if len(sent) > 0:
		# Get dictionary
		print(json.loads(sent[0]))
		# Create empty dataframe
		df = spark.createDataFrame(sc.emptyRDD, schema=schema)

		# Iterate and get new df
		# for i in d:
		# 	print(d[i])
		# print(d)
		# df.show()
	else:
		pass

if __name__ == "__main__":
	sc = SparkContext(appName="tweetStream")
	ssc = StreamingContext(sc, batchDuration= 3)
	spark = SparkSession.builder.getOrCreate()

	lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	words = lines.flatMap(lambda line : line.lower().split("\n"))
	
	words.foreachRDD(predict)

	# Start the computation
	ssc.start()             

	#wait till over
	ssc.awaitTermination(timeout=200)
