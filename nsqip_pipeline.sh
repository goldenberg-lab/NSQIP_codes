#!/bin/bash

# -------------------------------------- #
# Shell script to process the NSQIP data #
# -------------------------------------- #


conda activate nsqip

echo "----- Step 1: Process raw data files to csv or text -----"
python raw_nsqip_process.py
# Output: 