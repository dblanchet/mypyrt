#!/bin/sh

EXE_FILE=./mypyrt.py
STAT_FILE=stats.txt

# No subprocesses.
$EXE_FILE | tee $STAT_FILE

for subproc_count in {0..10}
do
    $EXE_FILE $subproc_count | tee -a $STAT_FILE
done
