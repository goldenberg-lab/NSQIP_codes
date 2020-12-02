#!/bin/bash

# -------------------------------------- #
# Shell script to process the NSQIP data #
# -------------------------------------- #
home=/hpf/largeprojects/agoldenb/ben
project=${home}/Projects/nsqip
test=${project}/NSQIP_codes


# Generate the patient data

for i in {2..10}; do # max_depth
  for j in 0.3 0.5 0.7 0.9; do # colsample_bytree
    echo "${test}/cpt_xgboost.py --m_depth $i --c_sample $j " | qsub -N "${i}_${j}" -l nodes=1:ppn=5,gres=localhd:1,vmem=10G,mem=10G,walltime=1:00:00:00 -e ${project}/Error
    sleep 0.1
  done
  sleep 0.1
done


