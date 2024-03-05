#!/bin/bash

python3 sarsa.py
wait
echo "Choose type of data folder base(0), long(1), none(2), slong(3)"
read NUM
VAL0=0
VAL1=1
VAL2=2
VAL3=3
echo "number: "
read NAME
FOLDER=9
if [ ${NUM} -eq ${VAL0} ]; then
        mv Log*.csv /home/ys/Desktop/Log/${FOLDER}/base/base${NAME}.csv
elif [ ${NUM} -eq ${VAL1} ]; then
	mv Log*.csv /home/ys/Desktop/Log/${FOLDER}/long/long${NAME}.csv
elif [ ${NUM} -eq ${VAL2} ]; then
	mv Log*.csv /home/ys/Desktop/Log/${FOLDER}/none/none${NAME}.csv
else
	mv Log*.csv /home/ys/Desktop/Log/${FOLDER}/slong/slong${NAME}.csv
fi
