#!/usr/bin/bash

gdown https://drive.google.com/uc?id=1vlczaocHyghEW5Q-6WprqB2nnna_wrLE

unzip stackoverflow.zip 

rm stackoverflow.zip

grep "python" questions.csv | wc

grep "python" question_tags.csv | wc
