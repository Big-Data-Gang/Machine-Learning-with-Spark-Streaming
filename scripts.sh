echo "starting server"
python3 stream.py -f sentiment -b 100
sleep 1
echo "starting client"
$SPARK_HOME/bin/spark-submit app.py  > output.txt
