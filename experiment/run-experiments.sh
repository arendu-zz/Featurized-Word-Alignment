SOURCE="toy.en"
TARGET="toy.fr"
KEY="toy.key"
MODEL="model1"

python ../convert_align.py $KEY toy.out.key
python initial_translation.py  -s $SOURCE -t $TARGET  -o initial.trans -m uniform
python convert_trans.py initial.trans

python ../featurized_em_wa.py -s $SOURCE -t $TARGET -a 'LBFGS' -m $MODEL --iw initial.trans.out
python ../convert_align.py LBFGS.$MODEL.alignments
echo ""
echo "*********LBFGS********"
echo ""
python eval_alignment.py toy.out.key LBFGS.$MODEL.alignments.out

python model1.py -s $SOURCE -t $TARGET -i initial.trans -p model1.probs -a model1.alignments -as $SOURCE -at $TARGET
echo ""
echo "********Baseline********"
echo ""
python eval_alignment.py toy.out.key model1.alignments
