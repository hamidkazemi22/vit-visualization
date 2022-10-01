#!/bin/bash
for l in $(seq 0 11) ; do
  echo "python reconstruct.py -l ${l} -f 0 -n 34"
  echo "python reconstruct.py -l ${l} -f 0 -n 35"
done
