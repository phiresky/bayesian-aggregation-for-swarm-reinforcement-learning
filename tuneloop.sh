#!/bin/bash
while true; do
	for i in {1..6}; do
		poetry run python -m playground.tune --db optuna.sqlite3 --study_name assembly-v01 &
		sleep 5
	done
	wait
	sleep 1m # in case something is wrong with above, prevent cpu usage from being so much
done
