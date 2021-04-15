#!/bin/bash

redis-cli FLUSHALL
flatland-evaluator --tests ./scratch/test-envs/ &
sleep 1
export AICROWD_TESTS_FOLDER=./scratch/test-envs/
python ./run.py &
