#! /usr/bin/python3

TCP_IP = "localhost"
TCP_PORT = 6100
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# ! preferably do in method 2

"""

 # --- way 1 ---
import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
sc = SparkContext(appName="tweetStream")
# Create a local StreamingContext with batch interval of 0.5 second
ssc = StreamingContext(sc, 0.5)
# Create a DStream that conencts to hostname:port
lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

# Split Tweets
words = lines.flatMap(lambda s: s.lower().split("__end"))
# Print the first ten elements of each DStream RDD to the console
# print("words.pprint() >> ", end = " ")
words.pprint()
# print("<< words.pprint() ", end = "||||||\n")
# Start computing
ssc.start()
# print("ssc.start() >>>>", end = " ")
# print(ssc)
# print("<<<< ssc.start()", end = " ------\n")
# Wait for termination
ssc.awaitTermination()
"""
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql import Row



def predict(tweet_text):
	# predict the sentiment of the tweet

	# the formatting isnt perfect, somewhat confusing
	tweet_text = tweet_text.replace("__end", "")
	# print(type(tweet_text))
	try:
		print(tweet_text.collect())
	except Exception as E: 
		print('failed', E)

if __name__ == "__main__":
	sc = SparkContext(appName="tweetStream")
	ssc = StreamingContext(sc, batchDuration= 3)
	spark = SparkSession(sc)

	lines = ssc.socketTextStream(TCP_IP, TCP_PORT)

	words = lines.flatMap(lambda line : line.lower().split("\n"))

	words.foreachRDD(predict)

	# Start the computation
	ssc.start()             

	#wait till over
	ssc.awaitTermination()  
