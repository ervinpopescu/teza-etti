#!/bin/bash

for i in "$@"
do
  ag -l . --ignore='*.ppm' --ignore='*.csv' --ignore='*scores.txt' | entr -c -s "make -j12 $i"
done &>/dev/null
