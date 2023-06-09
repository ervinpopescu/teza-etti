PYTHON=python -W ignore
DATE:=$(shell date +"%d/%m/%Y, %T")
RUN=${PYTHON} main.py $(1) &>/dev/null
PRINT_SCORES=scripts/print_scores.py |& tee -a main.log &>/dev/null
ECHO_DATE=echo "${GREEN}Start: ${BLUE}${DATE}{RESET}" > $(1)
# define standard colors
ifneq (,$(findstring xterm,${TERM}))
	BLACK        := $(shell tput -Txterm setaf 0)
	RED          := $(shell tput -Txterm setaf 1)
	GREEN        := $(shell tput -Txterm setaf 2)
	YELLOW       := $(shell tput -Txterm setaf 3)
	LIGHTPURPLE  := $(shell tput -Txterm setaf 4)
	PURPLE       := $(shell tput -Txterm setaf 5)
	BLUE         := $(shell tput -Txterm setaf 6)
	WHITE        := $(shell tput -Txterm setaf 7)
	RESET := $(shell tput -Txterm sgr0)
else
	BLACK        := ""
	RED          := ""
	GREEN        := ""
	YELLOW       := ""
	LIGHTPURPLE  := ""
	PURPLE       := ""
	BLUE         := ""
	WHITE        := ""
	RESET        := ""
endif

all: poc

clean:
	rm -rf output/*/{output*,scores.txt,predicted_labels.pickle,accuracies.txt}

check:
	$(call ECHO_DATE,main.log)
	$(call RUN)
	$(call PRINT_SCORES)
	eog output/plot.png &

train:
	$(call ECHO_DATE,main.log)
	$(call RUN,--train)

test:
	$(call ECHO_DATE,main.log)
	$(call RUN,--test)
	$(PRINT_SCORES)

test-images:
	$(call ECHO_DATE,main.log)
	$(call RUN,--test-images)
	$(PRINT_SCORES)

test-video:
	$(call ECHO_DATE,main.log)
	rm -f output/vgg16/output-50_speed.mp4 input/frames/vgg16/50_speed.frames.npy.gz
	$(call RUN,--test-videos)

test-videos:
	$(call ECHO_DATE,main.log)
	$(call RUN,--test-videos)

poc:
	$(call ECHO_DATE,poc.log)
	${PYTHON} poc.py &>/dev/null

.PHONY: all train test* clean
