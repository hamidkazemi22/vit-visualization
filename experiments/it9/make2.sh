#!/bin/bash
for f in $(seq 0 3071); do
  for l in $(seq 0 11); do
    echo "python vis.py -l ${l} -f ${f}"
  done
done
