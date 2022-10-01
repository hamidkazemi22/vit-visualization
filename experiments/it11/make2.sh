#!/bin/bash
for f in $(seq 0 3072) ; do
  for l in $(seq 0 11) ; do
    for v in 30.0 1.0; do
      echo "python vis_clip.py -l ${l} -v ${v} -f ${f}"
    done
  done
done