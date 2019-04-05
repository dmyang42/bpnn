#!/bin/zsh

for (( i=1; i <= 100; i++))
do
    python EEG_bpnn.py $i > train.log$i
done