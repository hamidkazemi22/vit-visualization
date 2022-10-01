#!/bin/bash
for l in $(seq 0 20); do
    echo "python bn.py -n 12 -l ${l} -f 0"
done
