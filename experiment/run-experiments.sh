#!/bin/sh
SOURCE_FULL="source_full.tmp"
TARGET_FULL="target_full.tmp"
SOURCE_TRAIN="data/toy.en"
TARGET_TRAIN="data/toy.fr"
SOURCE_TEST="data/toy.en"
TARGET_TEST="data/toy.fr"
KEY="data/toy.key"
MODEL="model1"
ALGO="EM"
echo "training files:" $SOURCE_TRAIN ","  $TARGET_TRAIN
echo "testing  files:" $SOURCE_TEST "," $TARGET_TEST
touch $SOURCE_FULL
touch $TARGET_FULL
cat $SOURCE_TEST > $SOURCE_FULL
cat $SOURCE_TRAIN >> $SOURCE_FULL
cat $TARGET_TEST > $TARGET_FULL
cat $TARGET_TRAIN >> $TARGET_FULL

python initial_translation.py  -s $SOURCE_FULL -t $TARGET_FULL  -o initial.trans -m uniform
python editdistance.py -i initial.trans > initial.feature.values
for ALGO in "LBFGS" "EM"
do
  for RC in 0.0 0.005
  do
    time python ../featurized_em_wa_mp.py -s $SOURCE_FULL -t $TARGET_FULL -a $ALGO -m $MODEL --iw initial.trans.log --ts $SOURCE_TEST --tt $TARGET_TEST -r $RC
  done
done



python model1.py -s $SOURCE_FULL -t $TARGET_FULL -i initial.trans -p model1.probs -a model1.alignments -as $SOURCE_TEST -at $TARGET_TEST
echo ""
echo "********Baseline********"
echo ""
python eval_alignment.py $KEY model1.alignments

for ALGO in "LBFGS" "EM"
do
    for RC in 0.0 0.005
    do
      echo ""
      echo "*********"$ALGO " RC:"$RC"********"
      echo ""
      python eval_alignment.py $KEY $ALGO.$RC.$MODEL.bin.alignments.col
    done
done
rm $SOURCE_FULL
rm $TARGET_FULL
