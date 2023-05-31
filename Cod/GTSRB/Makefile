PYTHON=python -W ignore

all: clean run

train:
	(time (${PYTHON} main.py --train |& tee main.log) | tee -a main.log) &>/dev/null

test:
	(time (${PYTHON} main.py --train --test |& tee main.log) | tee -a main.log) &>/dev/null
	scripts/print_scores.py |& tee -a main.log &>/dev/null

test-images:
	(time (${PYTHON} main.py --test-images |& tee main.log) | tee -a main.log) &>/dev/null
	scripts/print_scores.py |& tee -a main.log &>/dev/null

test-videos:
	(time (${PYTHON} main.py --test-videos |& tee -a main.log) | tee -a main.log) &>/dev/null

clean:
	rm -rf output/*/{output*,scores.txt,predicted_labels.pickle,accuracies.txt}

.PHONY: all train test* clean
