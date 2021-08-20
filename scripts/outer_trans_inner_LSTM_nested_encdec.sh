#!/bin/bash
for i in SG SA FC NF
do
    for ((j=0; j<10; j++))
    do
        ./inner_LSTM_loads_outer_transformer.py data/${i}-10-train.txt data/${i}-10-test.txt >> data/outer_trans_inner_LSTM_nested_query_${i}.txt
    done
done