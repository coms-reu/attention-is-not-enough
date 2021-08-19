#!/bin/bash
for i in SG SA FC NF
do
    for ((j=0; j<10; j++))
    do
        ./NestedTransformer_JLP.py ../../thesis-master/data/len5_10000-train.txt ../../thesis-master/data/${i}-10-train.txt ../../thesis-master/data/${i}-10-test.txt >> ../Results/NestedTransformer_WithTF/${i}_output.txt
    done
done