#!/bin/bash
for l in $(seq 2 3 11); do
  for r in 0.01 0.03 0.1 0.3 1.0 ; do
    for g in 10. 1. 0.1 0.01 ; do
      echo "python high.py -l ${l} -g ${g} -r ${r}"
    done
  done
done
