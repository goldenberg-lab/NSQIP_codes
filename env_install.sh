#!/bin/bash

e_nsqip=$(conda env list | grep NSQIP)
n_nsqip=$(echo $e_nsqip | wc -w)
n_nsqip=$(($n_nsqip))

echo "n_nsqip="$n_nsqip

if [ $n_nsqip -gt 0 ]; then
	echo "NSQIP environment already found"
else
	echo "NSQIP environment needs to be installed"
	# Loop through the pip packages
	fn="conda_env.txt"
	conda create --name NSQIP python=3.7
fi

conda activate NSQIP
echo "------------------- PYTHON LOCATION -------------------- "
which conda

fn="conda_env.txt" # filename with the different pip environments

# --- install/update packages --- #
n_line=$(cat $fn | grep -n pip | tail -1 | cut -d ":" -f1)
n_line=$(($n_line + 1))
n_end=$(cat $fn | grep -n prefix | tail -1 | cut -d ":" -f1)
n_end=$(($n_end - 1))
echo "line_start:"$n_line
echo "line_end:"$n_end
holder=""
for ii in `seq $n_line $n_end`; do
	echo "line: "$ii
	pckg=$(cat $fn | head -$ii | tail -1 | sed -e "s/[[:space:]]//g")
	pckg=$(echo $pckg | sed -e "s/^\-//g")
	holder=$holder" "$pckg
done
echo "packages: "$holder
pip install $holder

echo "--------------------------------------------------------"
echo "------------------ END OF SCRIPT -----------------------"
