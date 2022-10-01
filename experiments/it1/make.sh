#!/bin/bash
for l in `seq 0 20`; do
  for g in 0.0001 0.001 0.01 0.1 1.0 10. 100. 1000. ; do
    echo "python bn.py -n 12 -l ${l} -f 0 -g ${g}"
  done
done
