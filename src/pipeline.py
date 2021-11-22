from sparknlp import DocumentAssembler
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline
import json
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import Word2Vec as wv
class PreProcess:
    def __init__(self, df):
        self.df = df
        if self.df == None:
            return None
        self.preprocess()

    def regex(self):

        self.df = self.df.withColumn('tweet', F.regexp_replace('tweet', r'http\S+', ''))
        self.df = self.df.withColumn('tweet', F.regexp_replace('tweet', '@\w+', ''))
        self.df = self.df.withColumn('tweet', F.regexp_replace('tweet', '#', ''))
        self.df = self.df.withColumn('tweet', F.regexp_replace('tweet', 'RT', ''))
        self.df = self.df.withColumn('tweet', F.regexp_replace('tweet', ':', ''))

    
    def preprocess(self):
        # Cleanup mode is set to shrink

        self.regex()

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

        finisher = Finisher() \
            .setInputCols(["lemma"]) \
            .setIncludeMetadata(False) \
            .setOutputCols("finished")

        word2vec = wv() \
            .setInputCol("finished") \
            .setOutputCol("vector") \
            .setVectorSize(100) \
            .setMinCount(0)

        nlpPipeline = Pipeline(stages=[
            documentAssembler, 
            tokenizer,
            normalizer,
            # spellModel,
            stopwords_cleaner,
            stemmer,
            lemmatizer,
            finisher,
            word2vec,
        ])

        pipelineModel = nlpPipeline.fit(self.df)
        result = pipelineModel.transform(self.df)
        return result.select("sentiment", "vector")
