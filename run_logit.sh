#!/bin/bash

# -------------------------------------- #
# Shell script to process the NSQIP data #
# -------------------------------------- #
home=/hpf/largeprojects/agoldenb/ben
project=${home}/Projects/nsqip
test=${project}/NSQIP_codes


# Generate the patient data
for i in 0.01 0.1 1.0 10.0 100.00 1000.00; do
  echo $i
  echo "${test}/cpt_logit.py --c_value $i" | qsub -N "${i}" -l nodes=1:ppn=10,gres=localhd:1,vmem=10G,mem=10G,walltime=1:00:00:00 -o ${project}/jobs_output -e ${project}/Error
 sleep 0.5
done

