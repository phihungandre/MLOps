lint:
	pylint -disable... src/*.py

format:
	black -l 100 src/*.py

setup:
	pip install -r requirements.txt

clean_data:
	python src/features.py

train_model:
	python src/train.py

install:
	pip install --upgrade pip && pip install -r requirements.txt