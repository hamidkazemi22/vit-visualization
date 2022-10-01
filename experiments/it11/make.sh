#!/bin/bash
for f in $(seq 0 7) ; do
  for l in $(seq 0 11) ; do
    for v in 100.0 1.0; do
      echo "python vis_clip.py -l ${l} -v ${v} -f ${f}"
    done
  done
done