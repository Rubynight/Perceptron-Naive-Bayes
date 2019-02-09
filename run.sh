#!/bin/bash
for i in {1..10}; do
	python ./split_dataset.py 100
	echo $(python ./nbc.py train-set.csv test-set.csv)
	#echo $(python ./avg.py train-set.csv test-set.csv)
done
