.PHONY: init run

init:
	pip install --upgrade pip && pip install -r requirements.txt

run:
	python3 main.py
