1. Install the required files using `pip3 install -r requirements.txt`

2. Run `wget https://repo1.maven.org/maven2/com/johnsnowlabs/nlp/spark-nlp_2.12/3.3.2/spark-nlp_2.12-3.3.2.jar -P $SPARK_HOME/jars` to install the spark nlp jars package

3. Run these 2 commands from 2 terminals

        ```
        python3 src/stream.py -f sentiment -b 10000
        
        $SPARK_HOME/bin/spark-submit src/app.py > src/output.txt
        ```
