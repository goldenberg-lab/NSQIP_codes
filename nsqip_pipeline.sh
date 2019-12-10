#!/bin/bash

# -------------------------------------- #
# Shell script to process the NSQIP data #
# -------------------------------------- #


conda activate NSQIP

echo "----- Step 1: Process raw data files to csv or text -----"
python 01a_combine_data.py #raw_nsqip_process.py
# Output: 
