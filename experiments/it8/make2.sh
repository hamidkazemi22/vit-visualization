#!/bin/bash
for l in $(seq 0 2 11); do
    for g in 10. 1. 0.1 0.01 ; do
      echo "python cj_std.py -l ${l} -g ${g} -f 1"
  done
done
