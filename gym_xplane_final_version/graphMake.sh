#!/bin/bash

mv ~/GYM_XPLANE_ML/gym_xplane_final_version/*.csv ~/Desktop/Log/graph/log.csv
echo "Choose type of graph"
echo "--------------------------------"
echo "process time and perf count -- 0"
echo "process time ----------------- 1"
echo "perf count ------------------- 2"
read NUM
VAL0=0
VAL1=1
VAL2=2
if [ ${NUM} -eq ${VAL0} ]; then
	python3 ~/Desktop/Log/graph/graph_both.py
elif [ ${NUM} -eq ${VAL1} ]; then
	python3 ~/Desktop/Log/graph/graph_processtime.py
else
	python3 ~/Desktop/Log/graph/graph_perfcount.py
fi
	 
