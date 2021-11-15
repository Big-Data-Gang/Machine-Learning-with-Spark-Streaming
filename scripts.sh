echo "starting server"
python3 src/stream.py -f sentiment -b 1000
sleep 1
echo "starting client"
$SPARK_HOME/bin/spark-submit src/app.py > src/output.txt
