#!/bin/bash

# -------------------------------------- #
# Shell script to process the NSQIP data #
# -------------------------------------- #


conda activate NSQIP

echo "----- Step 1: Combine raw data from Stata format -----"
python 01a_combine_data.py 
# Output: combined_raw.csv and yr_vars.csv

echo "----- Step 2: Process data to produce y and X matrices -----"
python yX_process.py 
# Output: 
