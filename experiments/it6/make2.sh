#!/bin/bash
for l in $(seq 1 3 11); do
  for f in $(seq 0 5); do
    for m in in_feat keys queries values out_feat ; do
      echo "python head.py -l ${l} -f ${f} -m ${m}"
    done
  done
done
