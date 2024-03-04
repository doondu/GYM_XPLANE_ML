#!/bin/bash

python3 -m cProfile -o profile.pstats logExample.py
wait
gprof2dot -f pstats profile.pstats | dot -Tsvg -o callgraph.svg
wait
echo "Choose type of data folder base(0), long(1), none(2), slong(3)"
read NUM
VAL0=0
VAL1=1
VAL2=2
VAL3=3
echo "number: "
read NAME
FLODER=5
if [ ${NUM} -eq ${VAL0} ]; then
        mv Log*.csv /home/ys/Desktop/Log/${FOLDER}/base/base${NAME}.csv
	mv call*.svg /home/ys/Desktop/Log/${FOLDER}/base/call${NAME}.svg
elif [ ${NUM} -eq ${VAL1} ]; then
	mv Log*.csv /home/ys/Desktop/Log/${FOLDER}/long/long${NAME}.csv
	mv call*.svg /home/ys/Desktop/Log/${FOLDER}/long/call${NAME}.svg
elif [ ${NUM} -eq ${VAL2} ]; then
	mv Log*.csv /home/ys/Desktop/Log/${FOLDER}/none/none${NAME}.csv
	mv call*.svg /home/ys/Desktop/Log/${FOLDER}/none/call${NAME}.svg
else
	mv Log*.csv /home/ys/Desktop/Log/${FOLDER}/slong/slong${NAME}.csv
        mv call*.svg /home/ys/Desktop/Log/${FOLDER}/slong/call${NAME}.svg
fi
