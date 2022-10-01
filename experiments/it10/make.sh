#!/bin/bash
for l in $(seq 2 3 11) ; do
  for lr in 10.0 1.0 0.1 0.01 ; do
      for g in 100. 10. 1. 0.1 0.01 ; do
        echo "python param_tune.py -l ${l} -g ${g} -r ${lr}"
    done
  done
done