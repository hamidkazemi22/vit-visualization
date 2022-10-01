#!/bin/bash
lr=1.0
for l in $(seq 1 2 11) ; do
  for g in 100. 30. 10. 3.0 1. 0.3 0.1; do
    echo "python param_tune.py -l ${l} -g ${g} -r ${lr}"
  done
done