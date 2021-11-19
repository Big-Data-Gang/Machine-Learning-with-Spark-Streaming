import sparknlp
from sparknlp.base import *

# Start sparknlp
spark = sparknlp.start()
# Read Sample DF
df = spark.read.csv('./src/sentiment/sample.csv', header=True)
# df.show(truncate=False)

# Create document assembler to convert tweet to document type

documentAssembler = DocumentAssembler()\
                    .setInputCol('Tweet')\
                    .setOutputCol('doc')\
                    .setCleanupMode('shrink')

doc = documentAssembler.transform(df)

# See the result
# doc.select("doc.result").show(truncate=False)

# Sentence Assembler to detect sentences
