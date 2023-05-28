all: clean run

train:
	python -W ignore main.py --train

test:
	python -W ignore main.py --test

test-images:
	python -W ignore main.py --test-images

test-videos:
	python -W ignore main.py --test-videos

run:
	python -W ignore main.py --train --test

clean:
	rm -rf output/{output-*,scores.txt,predicted_labels.pickle}

.PHONY: all train test* clean
