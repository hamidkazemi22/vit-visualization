#!/bin/bash
for l in $(seq 0 2 11); do
    for g in 100. 10. 1. 0.1 0.01 ; do
      echo "python gaus.py -l ${l} -g ${g}"
  done
done
