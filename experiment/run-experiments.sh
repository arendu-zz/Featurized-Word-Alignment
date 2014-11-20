#!/bin/sh
SOURCE_FULL="source_full.tmp"
TARGET_FULL="target_full.tmp"
SOURCE_TRAIN="data/train.clean.tok.true.en"
TARGET_TRAIN="data/train.clean.tok.true.es"
SOURCE_TEST="data/dev.en"
TARGET_TEST="data/dev.es"
KEY="data/dev.key"
MODEL="model1"
touch $SOURCE_FULL
touch $TARGET_FULL
cat $SOURCE_TEST > $SOURCE_FULL
cat $SOURCE_TRAIN >> $SOURCE_FULL
cat $TARGET_TEST > $TARGET_FULL
cat $TARGET_TRAIN >> $TARGET_FULL

python initial_translation.py  -s $SOURCE_FULL -t $TARGET_FULL  -o initial.trans -m uniform

time python ../featurized_em_wa_mp.py -s $SOURCE_FULL -t $TARGET_FULL -a 'LBFGS' -m $MODEL --iw initial.trans.log --ts $SOURCE_TEST --tt $TARGET_TEST
echo ""
echo "*********LBFGS********"
echo ""
python eval_alignment.py $KEY LBFGS.$MODEL.alignments.col

python model1.py -s $SOURCE_FULL -t $TARGET_FULL -i initial.trans -p model1.probs -a model1.alignments -as $SOURCE_TEST -at $TARGET_TEST
echo ""
echo "********Baseline********"
echo ""
python eval_alignment.py $KEY model1.alignments

time python ../featurized_em_wa_mp.py -s $SOURCE_FULL -t $TARGET_FULL -a 'LBFGS' -m $MODEL --iw model1.probs --ts $SOURCE_TEST --tt $TARGET_TEST
echo ""
echo "*********LBFGS+model1********"
echo ""
python eval_alignment.py $KEY LBFGS.$MODEL.alignments.col

rm $SOURCE_FULL
rm $TARGET_FULL
