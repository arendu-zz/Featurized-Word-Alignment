SOURCE="data/dev.small.en"
TARGET="data/dev.small.es"
KEY="data/dev.small.key"
MODEL="model1"

python initial_translation.py  -s $SOURCE -t $TARGET  -o initial.trans -m uniform
python ../convert_trans.py initial.trans

python ../featurized_em_wa.py -s $SOURCE -t $TARGET -a 'LBFGS' -m $MODEL --iw initial.trans.out
echo ""
echo "*********LBFGS********"
echo ""
python eval_alignment.py $KEY LBFGS.$MODEL.alignments.col

python model1.py -s $SOURCE -t $TARGET -i initial.trans -p model1.probs -a model1.alignments -as $SOURCE -at $TARGET
echo ""
echo "********Baseline********"
echo ""
python eval_alignment.py $KEY model1.alignments
