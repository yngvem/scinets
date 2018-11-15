#!/bin/bash

for f in $(ls $1/experiments);
do
	python run_sacred.py db_credentials.yml $1/experiments/$f $2 --eval dice;
done;
