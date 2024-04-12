#!/usr/bin/env python3

import pandas as pd
import numpy as np
import time
from pyspark.sql.functions import when, rand
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    UniversalSentenceEncoder,
    SentimentDLApproach
)

import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU can be used.")
    print("Version: ", torch.version.cuda)
    print("Number of GPUs available: ", torch.cuda.device_count())
else:
    print("CUDA is not available. Using CPU instead.")

import sparknlp
import findspark as fs
fs.init('/home/jdu5sq/spark-3.4.1-bin-hadoop3')
fs.find()

print("Import done!")
data_path = "/home/jdu5sq/Documents/MSDS/DS5110/Project/"

from pyspark.sql import SparkSession

def start_spark_session():
    spark = SparkSession.builder \
        .appName("GPU Spark NLP") \
        .master("local[10]") \
        .config("spark.driver.memory", "16G") \
        .config("spark.executor.memory", "12G") \
        .config("spark.executor.instances", "4") \
        .config("spark.task.cpus", "1") \
        .config("spark.task.resource.gpu.amount", "0.25") \
        .config("spark.executor.resource.gpu.amount", "1") \
        .config("spark.executor.resource.gpu.discoveryScript", data_path+"/getGpusResources.sh") \
        .config("spark.driver.resource.gpu.amount", "1") \
        .config("spark.driver.resource.gpu.discoveryScript", data_path+"/getGpusResources.sh") \
        .getOrCreate()
    return spark

spark = start_spark_session()
sparknlp.start(gpu=True)

print("Spark started.")

print("Starting dataset making...")

schema = StructType([
    StructField("label", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("text", StringType(), True)
])

trainDataset = spark.read \
    .option("header", False) \
    .schema(schema) \
    .csv(data_path+"debugger_train.csv")

debugDataset = trainDataset.withColumn("label", when(trainDataset["label"] == 2, 1).otherwise(0))

print("Data loaded.")

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

sentimentdl = SentimentDLApproach() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("sentiment") \
    .setLabelColumn("label") \
    .setbatchSize(32) \
    .setlr(1e-3) \
    .setMaxEpochs(5) \
    .setEnableOutputLogs(True)

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

pipelineModel = pipeline.fit(debugDataset)

print("Model fitted.")

# End the timer
end_time = time.time()

# Calculate the total time taken
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")

# cat ~/annotator_logs/SentimentDLApproach_12faa854e3b3.log

print("Starting logs.")

# Define the log file path
log_file_path = '~/annotator_logs/SentimentDLApproach_12faa854e3b3.log'

# Print the contents of the log file
print("Contents of the log file:")
with open(log_file_path, 'r') as log_file:
    print(log_file.read())