#! /usr/bin/python3

TCP_IP = "localhost"
TCP_PORT = 6100

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql import Row
from sparknlp import DocumentAssembler
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline
import json
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.mllib.feature import Word2Vec


count = 0

def regex(df):

	df = df.withColumn('tweet', F.regexp_replace('tweet', r'http\S+', ''))
	df = df.withColumn('tweet', F.regexp_replace('tweet', '@\w+', ''))
	df = df.withColumn('tweet', F.regexp_replace('tweet', '#', ''))
	df = df.withColumn('tweet', F.regexp_replace('tweet', 'RT', ''))
	df = df.withColumn('tweet', F.regexp_replace('tweet', ':', ''))

	return df

def process(rdd):
	global count
	# Array of elements of the dataset
	sent = rdd.collect()

	if len(sent) > 0:
		df = spark.createDataFrame(data=json.loads(sent[0]).values(), schema=['sentiment', 'tweet'])
		df = preprocess(df)
		df.show(truncate=False)
		# count += df.count()
		# print(count)
		
		w2v = Word2Vec()
		model = w2v.fit(df.rdd)
		print(model)

def preprocess(df):
	# Cleanup mode is set to shrink

	df = regex(df)

	documentAssembler = DocumentAssembler()\
		.setInputCol("tweet")\
		.setOutputCol("document")\
		.setCleanupMode("shrink")

	tokenizer = Tokenizer() \
		.setInputCols(["document"]) \
		.setOutputCol("token")

	normalizer = Normalizer() \
		.setInputCols(["token"]) \
		.setOutputCol("normalized")\
		.setLowercase(True)\
		.setCleanupPatterns(["[^\w\d\s]"])

	# spellModel = ContextSpellCheckerModel\
	# 	.pretrained()\
	# 	.setInputCols("token")\
	# 	.setOutputCol("checked")\


	stopwords_cleaner = StopWordsCleaner()\
		.setInputCols("normalized")\
		.setOutputCol("cleanTokens")\
		.setCaseSensitive(False)

	stemmer = Stemmer() \
		.setInputCols(["cleanTokens"]) \
		.setOutputCol("stem")

	lemmatizer = Lemmatizer() \
		.setInputCols(["stem"]) \
		.setOutputCol("lemma") \
		.setDictionary("src/lemmas.txt", value_delimiter ="\t", key_delimiter = "->")

	# bert = BertSentenceEmbeddings.pretrained()\
	# 	.setInputCols('lemma')\
	# 	.setOutputCol('bert_embedding')

	nlpPipeline = Pipeline(stages=[
		documentAssembler, 
		tokenizer,
		normalizer,
		# spellModel,
		stopwords_cleaner,
		stemmer,
		lemmatizer,
		# bert
	])

	pipelineModel = nlpPipeline.fit(df)
	result = pipelineModel.transform(df)
	return result

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
