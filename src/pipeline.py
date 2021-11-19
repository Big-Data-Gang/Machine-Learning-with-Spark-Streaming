import sparknlp

# Start sparknlp
spark = sparknlp.start()
# Read Sample DF
df = spark.read.csv('./src/sentiment/sample.csv')
df.show(truncate=False)