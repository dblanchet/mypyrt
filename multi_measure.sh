#!/bin/sh

EXE_FILE=./mypyrt.py
STAT_FILE=stats.txt

# 0 stands for no subprocesses.
for subproc_count in {0..10}
do
    $EXE_FILE -j $subproc_count -o result$subproc_count.png | tee -a $STAT_FILE
done
