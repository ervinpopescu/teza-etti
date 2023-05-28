#!/bin/awk -f

{ total += $1; count++ } 
END {printf "Number of runs: %d\n",count ;printf "Average accuracy: %.2f%%", total/count }
