#!/bin/bash
set -u
simultaneous=$1; shift
iterations=$1; shift

cd "$(dirname "$0")"

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo running $iterations iterations, $simultaneous simultaneous of "$@"
for i in $(seq 1 $iterations); do
	for j in $(seq 1 $simultaneous); do
		echo poetry run "$@"
		poetry run "$@-run$i.$j" &
		sleep 60
	done
	wait
	sleep 5
done
