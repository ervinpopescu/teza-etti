#!/bin/python

import glob
import os
import pathlib

RED = "\033[1;31m"
GREEN = "\033[1;32m"
BLUE = "\033[1;34m"
RESET = "\033[0m"
cwd = pathlib.Path(__file__).parent.parent.resolve()
scores_files = sorted(glob.glob(os.path.join(cwd, "output", "scores*.txt")))
accuracy_files = sorted(glob.glob(os.path.join(cwd, "output", "accuracies*.txt")))
print()
for score_file, accuracy_file in zip(scores_files, accuracy_files):
    print(
        f"{BLUE}Base model architecture:",
        score_file.split("/")[-1].split("_")[-1].split(".")[0],
        "\n",
    )
    with open(score_file, "r") as f:
        print(f"\t{RED}All images test{RESET}\n")
        for line in f.readlines():
            print(f"\t\t{GREEN}{line.strip()}")
    with open(accuracy_file, "r") as f:
        print(f"\n\t{RED}Random images test{RESET}\n")
        lines = [float(x.strip()) for x in f.readlines()]
        print(
            f"\t\t{GREEN}Random image accuracy: {sum(lines)/len(lines):.2f}%"
            if len(lines) != 0
            else "\t\tRandom image accuracy: 0%",
            "\n",
        )
