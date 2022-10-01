#!/bin/bash
for f in $(seq 0 256); do
  for l in $(seq 0 11); do
    for v in 0.1 1.0 ; do
      echo "python vis35.py -l ${l} -f ${f} -v ${v} -r 0.1 -n 35"
      echo "python vis98.py -l ${l} -f ${f} -v ${v} -r 0.1 -n 98"
    done
  done
done
