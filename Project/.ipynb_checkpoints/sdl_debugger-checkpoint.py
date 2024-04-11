#!/usr/bin/env python3

import pandas as pd
import numpy as np
import time
from pyspark.sql.functions import when
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    UniversalSentenceEncoder,
    SentimentDLApproach
)

import sparknlp
import findspark as fs
fs.init('/home/jdu5sq/spark-3.4.1-bin-hadoop3')
fs.find()

print("Import done!")

params = {
    "spark.driver.cores":"4",
    "spark.driver.memory":"8G",
    "spark.executor.memory":"8G",
    "spark.master":"local[4]"
}
spark = sparknlp.start(gpu=True, params=params)

print("Spark started.")

data_path = "Documents/MSDS/DS5110/Project/"
trainDataset = spark.read \
      .option("header", False) \
      .csv(data_path+"train.csv")

header_names = ["label", "title", "text"]
trainDataset = trainDataset.toDF(*header_names)
trainDataset = trainDataset.withColumn("label", when(trainDataset["label"] == 2, 1).otherwise(0))

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

pipelineModel = pipeline.fit(trainDataset)

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