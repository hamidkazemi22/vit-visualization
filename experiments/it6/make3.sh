#!/bin/bash
for l in $(seq 0 11); do
  for f in $(seq 0 5); do
    echo "python score.py -l ${l} -f ${f}"
  done
done
