#!/bin/bash
for f in $(seq 0 3071); do
  for l in $(seq 0 11); do
    for v in 1. 10.0 ; do
      echo "python vis38.py -l ${l} -f ${f} -v ${v}"
    done
  done
done
