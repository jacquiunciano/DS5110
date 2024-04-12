#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "{\"name\": \"gpu\", \"addresses\":[$(seq -s , 0 $((NUM_GPUS-1)))]}"