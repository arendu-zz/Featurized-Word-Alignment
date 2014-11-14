#!/bin/sh
CORPUS_FOLesR="data/small"
CORPUS_PREFIX="20"
${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -l en < ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.en > ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.tok.en
${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -l es < ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.es > ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.tok.es

#learn truecase model
${MOSES_DIR}/scripts/recaser/train-truecaser.perl --model ${PROJECT_DIR}/${CORPUS_FOLesR}/truecase-model.en --corpus  ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.tok.en
${MOSES_DIR}/scripts/recaser/train-truecaser.perl --model ${PROJECT_DIR}/${CORPUS_FOLesR}/truecase-model.es --corpus  ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.tok.es

#do truecasing
${MOSES_DIR}/scripts/recaser/truecase.perl --model ${PROJECT_DIR}/${CORPUS_FOLesR}/truecase-model.en < ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.tok.en > ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.tok.true.en
${MOSES_DIR}/scripts/recaser/truecase.perl --model ${PROJECT_DIR}/${CORPUS_FOLesR}/truecase-model.es < ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.tok.es > ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.tok.true.es

#do cleaning for both en and es
${MOSES_DIR}/scripts/training/clean-corpus-n.perl ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.tok.true es en ${PROJECT_DIR}/${CORPUS_FOLesR}/${CORPUS_PREFIX}.clean.tok.true 1 80
