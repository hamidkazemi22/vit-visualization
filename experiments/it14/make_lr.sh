#!/bin/bash
for l in $(seq 2 3 11); do
  for r in 0.01  0.1  1.0 ; do
    for v in 10. 1. 0.1 0.01 ; do
      echo "python lr_rate.py -l ${l} -v ${v} -r ${r} -n 35"
    done
  done
done
