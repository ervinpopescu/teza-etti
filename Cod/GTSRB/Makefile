all: clean run

train:
	python main.py --train

test:
	python main.py --train --test
	scripts/print_scores.py

test-images:
	python main.py --test-images
	scripts/print_scores.py

test-videos:
	python main.py --test-videos

clean:
	rm -rf output/*/{output*,scores*.txt,predicted_labels*.pickle,accuracies*.txt}

.PHONY: all train test* clean
