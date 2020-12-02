#!/bin/bash

# -------------------------------------- #
# Shell script to process the NSQIP data #
# -------------------------------------- #
home=/hpf/largeprojects/agoldenb/ben
project=${home}/Projects/nsqip
test=${project}/NSQIP_codes


# Generate the patient data

for i in {2..10}; do # max_depth
  for j in 20.0 50.0 100.0 500.0 1000.0; do # colsample_bytree
    echo "${test}/cpt_rf.py --m_depth $i --n_est $j " | qsub -N "${i}_${j}" -l nodes=1:ppn=5,gres=localhd:1,vmem=10G,mem=10G,walltime=1:00:00:00 -o ${project}/jobs_output -e ${project}/Error
    sleep 0.1
  done
  sleep 0.1
done


