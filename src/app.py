#! /usr/bin/python3

TCP_IP = "localhost"
TCP_PORT = 6100

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql import Row
from sparknlp import DocumentAssembler
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import json
import pkg_resources
from symspellpy import SymSpell

count = 0

def spellCheck(text, dict_path):
	sym_spell = SymSpell()
	dictionary_path = pkg_resources.resource_filename(
	    	"symspellpy", 
		dict_path
	)
	sym_spell.load_dictionary(dictionary_path, 0, 1)
	input_term = None
	for i in text:
		try:
		
			input_term = i
			
			# max edit distance per lookup
			# (max_edit_distance_lookup <= max_dictionary_edit_distance)
			suggestions = sym_spell.lookup(input_term, Verbosity.CLOSEST,
				               max_edit_distance=2, include_unknown=True)
			# display suggestion term, term frequency, and edit distance
			text.replace(i, suggestions[0])
		except:
			continue
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
		
def preprocess(df):
	# Cleanup mode is set to shrink
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

	nlpPipeline = Pipeline(stages=[
		documentAssembler, 
		tokenizer,
		normalizer,
		stopwords_cleaner,
		stemmer,
		lemmatizer
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
	words = lines.flatMap(lambda line : line.lower().split("\n"))
	
	words.foreachRDD(process)
	
	
	# Start the computation
	ssc.start()
	
	#wait till over
	ssc.awaitTermination()
	ssc.stop(stopGraceFully=True)
