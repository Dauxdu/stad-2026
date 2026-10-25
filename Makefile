.PHONY: init run

init:
	pip install --upgrade pip && pip install -r requirements.txt

run:
	python3 main.py data/var_01.csv --k-max 50 --orders 2 20
