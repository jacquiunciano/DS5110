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
        .config("spark.task.resource.gpu.amount", "0.5") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.resource.gpu.amount", "1") \
        .config("spark.executor.resource.gpu.discoveryScript", data_path+"/getGpusResources.sh") \
        .config("spark.driver.resource.gpu.amount", "1") \
        .config("spark.driver.resource.gpu.discoveryScript", data_path+"/getGpusResources.sh") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0,com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:4.2.0") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.driver.maxResultSize", "0") \
        .getOrCreate()
    return spark

spark = start_spark_session()

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

# Start the timer
start_time = time.time()

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
print("documentAssembler finished!")
da_time = time.time() - start_time

useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")
print("useEmbeddings finished!")
ue_time = time.time() - da_time

sentimentdl = SentimentDLApproach() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("sentiment") \
    .setLabelColumn("label") \
    .setbatchSize(32) \
    .setlr(1e-3) \
    .setMaxEpochs(5) \
    .setEnableOutputLogs(True)
print("sentimentdl finished!")
dla_time = time.time() - ue_time

pipeline = Pipeline() \
    .setStages(
      [
        documentAssembler,
        useEmbeddings,
        sentimentdl
      ]
    )

print("Pipeline finished!")
pipe_time = time.time() - dla_time

pipelineModel = pipeline.fit(debugDataset)

print("Model fitted.")

# End the timer
end_time = time.time()

# Calculate the total time taken
total_time = end_time - start_time
print(f"Total DocumentAssembler time: {da_time} seconds")
print(f"Total UniversalSentenceEncoder time: {ue_time} seconds")
print(f"Total SentimentDLApproach time: {dla_time} seconds")
print(f"Total Pipline time: {pip_time} seconds")
print(f"Total execution time: {total_time} seconds")

# cat ~/annotator_logs/SentimentDLApproach_12faa854e3b3.log

print("Starting logs.")

# Define the log file path
log_file_path = '~/annotator_logs/SentimentDLApproach_12faa854e3b3.log'

# Print the contents of the log file
print("Contents of the log file:")
with open(log_file_path, 'r') as log_file:
    print(log_file.read())