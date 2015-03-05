#!/bin/sh
SOURCE_FULL="source_full.tmp"
TARGET_FULL="target_full.tmp"
SOURCE_TRAIN="data/train.en"
TARGET_TRAIN="data/train.es"
SOURCE_TEST="data/dev.en"
TARGET_TEST="data/dev.es"
KEY="data/dev.key"
DICT_PATH="data/dictionary_features.es-en"
MODEL="hybrid-model1"
echo "training files:" $SOURCE_TRAIN ","  $TARGET_TRAIN
echo "testing  files:" $SOURCE_TEST "," $TARGET_TEST
touch $SOURCE_FULL
touch $TARGET_FULL
cat $SOURCE_TEST > $SOURCE_FULL
cat $SOURCE_TRAIN >> $SOURCE_FULL
cat $TARGET_TEST > $TARGET_FULL
cat $TARGET_TRAIN >> $TARGET_FULL
rm mp.*
rm sp.*
python initial_translation.py  -s $SOURCE_FULL -t $TARGET_FULL  -o initial.trans -m uniform
time python model1.py -s $SOURCE_FULL -t $TARGET_FULL -i initial.trans -p model1.probs -a model1.alignments -as $SOURCE_TEST -at $TARGET_TEST

#python editdistance.py -i initial.trans > initial.feature.values
for ALGO in "LBFGS"  
do
    for RC in 0.005 
    do
        #time python ../hybrid_model1.py -s $SOURCE_FULL -t $TARGET_FULL -a $ALGO  --m1 model1.probs --ts $SOURCE_TEST --tt $TARGET_TEST -r $RC 
        time python ../hybrid_model1.py -s $SOURCE_FULL -t $TARGET_FULL -a $ALGO  --m1 model1.probs --ts $SOURCE_TEST --tt $TARGET_TEST -r $RC --df $DICT_PATH
        echo "."
    done
done


echo ""
echo "********Baseline********"
echo ""
python eval_alignment.py $KEY model1.alignments

for ALGO in "LBFGS" 
do
    for RC in 0.005
    do
        echo ""
        echo "*********SIMPLE "$ALGO " RC:"$RC"********"
        echo ""
        python eval_alignment.py $KEY sp.$ALGO.$RC.$MODEL.bin.alignments.col
    done
done
rm $SOURCE_FULL
rm $TARGET_FULL 
