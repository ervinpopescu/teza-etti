#!/bin/bash

# RED='\033[0;31m'
# NC='\033[0m'
# output="$(printf "%b" "${NC}${RED}-------\noutput\n-------\n${NC}"; ffprobe -v quiet -print_format json -show_format -show_streams "$1" | jq --color-output; printf "%b" "\n${NC}${RED}------\ninput\n------\n${NC}"; ffprobe -v quiet -print_format json -show_format -show_streams "$2" | jq --color-output)"
output="$(ffprobe -v quiet -print_format json -show_format -show_streams "$1" | jq -S)"
printf "%s" "$output"
