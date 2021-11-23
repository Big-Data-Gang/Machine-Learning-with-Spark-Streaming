echo "starting server"
cd src
python3 stream.py -f sentiment -b 1000&
sleep 1
echo "starting client"
cd ..
$SPARK_HOME/bin/spark-submit src/app.py > src/output.txt

