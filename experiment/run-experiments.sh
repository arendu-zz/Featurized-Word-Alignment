SOURCE="data/dev.small.en"
TARGET="data/dev.small.es"
KEY="data/dev.small.key"
MODEL="model1"

python initial_translation.py  -s $SOURCE -t $TARGET  -o initial.trans -m uniform

time python ../featurized_em_wa_mp.py -s $SOURCE -t $TARGET -a 'LBFGS' -m $MODEL --iw initial.trans.log
echo ""
echo "*********LBFGS********"
echo ""
python eval_alignment.py $KEY LBFGS.$MODEL.alignments.col

python model1.py -s $SOURCE -t $TARGET -i initial.trans -p model1.probs -a model1.alignments -as $SOURCE -at $TARGET
echo ""
echo "********Baseline********"
echo ""
python eval_alignment.py $KEY model1.alignments
