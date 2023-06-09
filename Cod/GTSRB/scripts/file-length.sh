#!/bin/sh

wc -l "$1" | awk '{print "\\newcommand\\Lines{"$1"}"}'
