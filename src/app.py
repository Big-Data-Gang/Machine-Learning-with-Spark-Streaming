#! /usr/bin/python3

TCP_IP = "localhost"
TCP_PORT = 6100

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row



def predict(tweet_text):
	
	sent = tweet_text.toDF()
	print(sent)

if __name__ == "__main__":
	sc = SparkContext(appName="tweetStream")
	ssc = StreamingContext(sc, batchDuration= 3)


	lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	words = lines.flatMap(lambda line : line.lower().split("\n"))
	
	words.foreachRDD(predict)

	# Start the computation
	ssc.start()             

	#wait till over
	ssc.awaitTermination(timeout=200)
