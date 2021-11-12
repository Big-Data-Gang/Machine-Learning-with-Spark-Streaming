from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Creating a Spark Context
sc = SparkContext()
# Creating a Spark Streaming Context
stream = StreamingContext(sparkContext= sc, batchDuration=1)
# Creating a DStream
text = stream.socketTextStream('localhost', 6100)

# Print
text.pprint()

# Start and stop
stream.start()
stream.awaitTerminationOrTimeout()