#!/usr/bin/env python3

import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU can be used.")
    print("Version: ", torch.version.cuda)
    print("Number of GPUs available: ", torch.cuda.device_count())
else:
    print("CUDA is not available. Using CPU instead.")
    
import pandas as pd
import numpy as np
import time
import os
from pyspark.sql.functions import when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

import sparknlp
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    UniversalSentenceEncoder,
    SentimentDLApproach
)

import sys

print("Before setting spark home:")
os.system("echo $SPARK_HOME")

os.environ['SPARK_HOME'] = '/home/jdu5sq/spark-3.4.1-bin-hadoop3'

print("After setting spark home:")
os.system("echo $SPARK_HOME")

os.system("wget -N https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/spell/words.txt -P /tmp")
os.system("rm -rf /tmp/sentiment.parquet")
os.system("wget -N https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sentiment.parquet.zip -P /tmp")
os.system("unzip /tmp/sentiment.parquet.zip -d /tmp/")

print("Import and downloads finished.")

spark = sparknlp.start(gpu=True, memory="32G")

print("Spark initialised.")

print("Starting dataset making...")

schema = StructType([
    StructField("label", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("text", StringType(), True)
])

trainDataset = spark.read \
    .option("header", False) \
    .schema(schema) \
    .csv("/home/jdu5sq/Documents/MSDS/DS5110/Project/train.csv")

print("Finished getting dataset.")

trainDataset = trainDataset.withColumn("label", when(trainDataset["label"] == 2, 1).otherwise(0))

spark.sparkContext.setLogLevel("ERROR")

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
print("documentAssembler finished!")

useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")
print("useEmbeddings finished!")

sentimentdl = SentimentDLApproach() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("sentiment") \
    .setLabelColumn("label") \
    .setBatchSize(32) \
    .setLr(1e-3) \
    .setMaxEpochs(5) \
    .setEnableOutputLogs(True)
print("sentimentdl finished!")

pipeline = Pipeline() \
    .setStages(
      [
        documentAssembler,
        useEmbeddings,
        sentimentdl
      ]
    )

print("Pipeline finished!")

# Start the timer
start_time = time.time()

pipelineModel = pipeline.fit(trainDataset)

print("Model fitted.")

# End the timer
end_time = time.time()

# Calculate the total time taken
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")

# cat ~/annotator_logs/SentimentDLApproach_12faa854e3b3.log

print("Starting logs.")

os.system('rm -r ~/annotator_logs/*')
os.system('cat ~/annotator_logs/SentimentDLApproach_*.log')

print("Logs finished.")

spark.stop()