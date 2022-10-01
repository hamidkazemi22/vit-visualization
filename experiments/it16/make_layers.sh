#!/bin/bash 
for l in $(seq 0 12) ; do
	for n in 34 35 ; do
	  echo "python layer_accuracy.py -l ${l} -n ${n}"
	done
done
