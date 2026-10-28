.PHONY: help setup lint test data baseline train evaluate export report all clean

PY ?= python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## install the package and dev tooling
	$(PY) -m pip install -e ".[dev]"

lint:  ## static checks
	ruff check src tests

test:  ## unit tests
	pytest

data:  ## generate the IQ datasets (~1.5 min, ~360 MB)
	rfmc data

baseline:  ## fit and score the cumulant + gradient-boosting baseline
	rfmc baseline

train:  ## fit the CNN
	rfmc train

evaluate:  ## score the CNN on the test split
	rfmc evaluate

export:  ## ONNX export, int8 quantisation, latency benchmark
	rfmc export

report:  ## collect metrics and figures into reports/
	rfmc report

all: data baseline train evaluate export report  ## full reproduction

clean:
	rm -rf data models reports/*.json reports/*.png reports/RESULTS.md
