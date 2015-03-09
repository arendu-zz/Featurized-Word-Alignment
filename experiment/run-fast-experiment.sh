#!/bin/sh
SOURCE_FULL="source_full.tmp"
TARGET_FULL="target_full.tmp"
SOURCE_TRAIN="data/train.en"
TARGET_TRAIN="data/train.es"
SOURCE_TEST="data/dev.en"
TARGET_TEST="data/dev.es"
KEY="data/dev.key"
MODEL="fast-model1"
echo "training files:" $SOURCE_TRAIN ","  $TARGET_TRAIN
echo "testing  files:" $SOURCE_TEST "," $TARGET_TEST
touch $SOURCE_FULL
touch $TARGET_FULL
cat $SOURCE_TEST > $SOURCE_FULL
cat $SOURCE_TRAIN >> $SOURCE_FULL
cat $TARGET_TEST > $TARGET_FULL
cat $TARGET_TRAIN >> $TARGET_FULL

python initial_translation.py  -s $SOURCE_FULL -t $TARGET_FULL  -o initial.trans
#python editdistance.py -i initial.trans > initial.feature.values
for ALGO in "EM-SGD-PARALLEL" 
do
    for RC in 0.0 
    do
        #time python ../featurized_fast_align_mp.py -s $SOURCE_FULL -t $TARGET_FULL -a $ALGO  --iw initial.trans --ts $SOURCE_TEST --tt $TARGET_TEST -r $RC
        echo "."
    done
done


time python model1.py -s $SOURCE_FULL -t $TARGET_FULL -i initial.trans -p model1.probs -a model1.alignments --as $SOURCE_TEST --at $TARGET_TEST
echo ""
echo "********Baseline********"
echo ""
python eval_alignment.py $KEY model1.alignments
for MODEL in "fast-model1"
do
for ALGO in "EM" 
do
    for RC in 0.0 
    do
        echo ""
        echo "********* MODEL:"$MODEL" ALGO:"$ALGO " RC:"$RC"********"
        echo ""
        python eval_alignment.py $KEY mp.$ALGO.$RC.$MODEL.bin.alignments.col
    done
done
done
rm $SOURCE_FULL
rm $TARGET_FULL 
