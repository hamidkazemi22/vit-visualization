#!/bin/bash
for l in $(seq 0 11); do
  for f in $(seq 0 5); do
    for g in 10. 1. 0.1 0.01 ; do
      echo "python best.py -l ${l} -f ${f} -g ${g}"
    done
  done
done
