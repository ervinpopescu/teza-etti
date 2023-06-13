MAKEFLAGS += --silent
# define standard colors
ifneq (,$(findstring alacritty,${TERM}))
	BLACK        := $(shell tput -Txterm setaf 0)
	RED          := $(shell tput -Txterm setaf 1)
	GREEN        := $(shell tput -Txterm setaf 2)
	YELLOW       := $(shell tput -Txterm setaf 3)
	LIGHTPURPLE  := $(shell tput -Txterm setaf 4)
	PURPLE       := $(shell tput -Txterm setaf 5)
	BLUE         := $(shell tput -Txterm setaf 6)
	WHITE        := $(shell tput -Txterm setaf 7)
	RESET 		 := $(shell tput -Txterm sgr0)
else ifneq (,$(findstring tmux,${TERM}))
	BLACK        := $(shell tput -Txterm setaf 0)
	RED          := $(shell tput -Txterm setaf 1)
	GREEN        := $(shell tput -Txterm setaf 2)
	YELLOW       := $(shell tput -Txterm setaf 3)
	LIGHTPURPLE  := $(shell tput -Txterm setaf 4)
	PURPLE       := $(shell tput -Txterm setaf 5)
	BLUE         := $(shell tput -Txterm setaf 6)
	WHITE        := $(shell tput -Txterm setaf 7)
	RESET        := $(shell tput -Txterm sgr0)
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
PYTHON=python -W ignore
DATE:=$(shell date +"%d/%m/%Y, %T")
RUN_MAIN=${PYTHON} main.py ${PROG_FLAGS} $(1)
RUN_POC=${PYTHON} poc.py ${PROG_FLAGS} $(1)
ECHO=echo $(GREEN)$$ $(BLUE)$(1)$(WHITE)
EOG=eog output/plot.png

all: help

help:
	$(call ECHO,$(call RUN_MAIN,--help))
	$(call RUN_MAIN,--help)

clean:
	$(call ECHO,rm -rf output/*/*)
	rm -rf output/*/*

train:
	$(call ECHO,$(call RUN_MAIN))
	$(call RUN_MAIN)
	# $(call ECHO,$(call EOG))
	# $(call EOG)

test:
	$(call ECHO,$(call RUN_MAIN,--test))
	$(call RUN_MAIN,--test)

test-images:
	$(call ECHO,$(call RUN_MAIN,--test-images))
	$(call RUN_MAIN,--test-images) && ($(call ECHO,$(call PRINT_SCORES)); $(call PRINT_SCORES))

test-video:
	$(call ECHO,rm -f output/vgg16/output-50_speed.mp4 input/frames/vgg16/50_speed.frames.npy.gz)
	rm -f output/vgg16/output-50_speed.mp4 input/frames/vgg16/50_speed.frames.npy.gz
	$(call ECHO,$(call RUN_MAIN,--test-videos))
	$(call RUN_MAIN,--test-videos)

test-videos:
	$(call ECHO,$(call RUN_MAIN,--test-videos))
	$(call RUN_MAIN,--test-videos)

poc:
	$(call ECHO,$(call RUN_POC))
	$(call RUN_POC)

.PHONY: all clean train test*
