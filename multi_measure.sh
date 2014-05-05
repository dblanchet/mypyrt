#!/bin/sh

EXE_FILE=./mypyrt.py
STAT_FILE=stats.txt

# 0 stands for no subprocesses.
for subproc_count in {0..10}
do
    $EXE_FILE $subproc_count | tee -a $STAT_FILE
done
