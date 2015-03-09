__author__ = 'arenduchintala'

"""
parameters in model 1:
delta[k,i,j] = translation[ foreign[k,i], english[k,j]] / sum_j_0toL (translation( foreign[k,i], english[k,j]))
k = kth sentences in the corpus
i = ith word in the kth sentence in the foreign corpus
j = jth word in the kth sentence in the english corpus
L = total number of words in the kth sentence in the english corpus
M = total number of words in the kth sentence in the foreign corpus
"""
"""
counts in model 1:
count[ejk, fik] = count[ejk, fik] + delta[k,i,j]
count[ejk] = count[ejk] + delta[k,i,j]
"""
"""
translation
translation[f,e] = c(f,e) / c(e)
"""
"""
for debugging purposes
https://class.coursera.org/nlangp-001/forum/thread?thread_id=940#post-4052
"""
import numpy as np
from optparse import OptionParser
import sys


np.set_printoptions(precision=4, linewidth=180)


def display_best_alignment(ak, en, es):
    lk = len(en)
    mk = len(es)
    k_mat = np.zeros((mk, lk))
    for jk in range(lk):
        for ik in range(mk):
            k_mat[ik][jk] = delta[ak, ik, jk]
    print ' '.join(en)
    print ' '.join(es)
    for ik, max_jk in enumerate(np.argmax(k_mat, 1)):
        print ik, max_jk, corpus_target[ak][ik], corpus_source[ak][max_jk]


if __name__ == "__main__":
    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/dev.es")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/dev.en")
    opt.add_option("--at", dest="target_test", default=None)
    opt.add_option("--as", dest="source_test", default=None)
    opt.add_option("-i", dest="initial_trans", default="experiment/data/init.trans")
    opt.add_option("-p", dest="save_trans", default="experiment/data/model1.probs")
    opt.add_option("-a", dest="ali_out", default="experiment/data/model1.alignment")
    (options, _) = opt.parse_args()

    source = options.source_corpus
    target = options.target_corpus
    ali_target = options.target_test if options.target_test is not None else options.target_corpus
    ali_source = options.source_test if options.source_test is not None else options.source_corpus
    save_trans = options.save_trans
    ali_out = options.ali_out
    init_translation = options.initial_trans

    delta = {}
    translations = {}
    counts = {}
    corpus_source = open(source, 'r').readlines()
    corpus_target = open(target, 'r').readlines()
    init_translation = open(init_translation, 'r').readlines()
    # corpus_source = corpus_source[:100]
    # corpus_target = corpus_target[:100]

    for line in init_translation:
        if line[0] is '#':
            pass  # it is a comment
        else:
            [emission, t, s, p] = line.split()
            translations[t, s] = np.exp(float(p))

    """
    EM iterations
    """
    for iter in range(5):
        counts = dict.fromkeys(counts.iterkeys(), 0.0)
        """
        accumilate fractional counts, E-Step
        """
        for k, source_sentence in enumerate(corpus_source):
            target_sentence = corpus_target[k]
            source_tokens = source_sentence.split()
            source_tokens.insert(0, 'NULL')
            target_tokens = target_sentence.split()
            t_mat = np.zeros((len(target_tokens), len(source_tokens)))
            for j in range(0, len(source_tokens)):
                for i in range(0, len(target_tokens)):
                    t_mat[i][j] = translations[target_tokens[i], source_tokens[j]]
            t_sum = np.sum(t_mat, 1)

            for j in range(0, len(source_tokens)):
                for i in range(0, len(target_tokens)):
                    delta[k, i, j] = t_mat[i][j] / t_sum[i]
                    counts[target_tokens[i], source_tokens[j]] = counts.get((target_tokens[i], source_tokens[j]), 0.0) + \
                                                                 delta[k, i, j]
                    counts[source_tokens[j]] = counts.get(source_tokens[j], 0.0) + delta[k, i, j]

        """
        update translations, M-Step
        """
        for target_i, source_j in translations:
            translations[target_i, source_j] = counts[target_i, source_j] / counts[source_j]

    TYPE = "EMISSION"
    writer = open(save_trans, 'w')
    for k in sorted(translations):
        v = translations[k]
        writer.write(TYPE + '\t' + str('\t'.join(k)) + '\t' + str(v) + '\n')
    writer.flush()
    writer.close()

    writer = open(ali_out, 'w')
    test_source = open(ali_source, 'r').readlines()
    test_target = open(ali_target, 'r').readlines()
    for dk in range(len(test_source)):
        source_tokens = test_source[dk].split()
        source_tokens.insert(0, 'NULL')
        target_tokens = test_target[dk].split()
        for i, token_target in enumerate(target_tokens):
            max_p = 0.0
            max_j = 0.0
            for j, token_source in enumerate(source_tokens):
                if translations[token_target, token_source] > max_p:
                    max_p = translations[token_target, token_source]
                    max_j = j
            if max_j > 0:
                writer.write(str(dk + 1) + ' ' + str(max_j) + ' ' + str(i + 1) + '\n')
    writer.flush()
    writer.close()
    """
    writer = open(ali_out + '.token', 'w')
    test_source = open(ali_source, 'r').readlines()
    test_target = open(ali_target, 'r').readlines()
    for dk in range(len(test_source)):
        source_tokens = test_source[dk].split()
        source_tokens.insert(0, 'NULL')
        target_tokens = test_target[dk].split()
        for i, token_target in enumerate(target_tokens):
            max_p = 0.0
            max_j = 0.0
            for j, token_source in enumerate(source_tokens):
                if translations[token_target, token_source] > max_p:
                    max_p = translations[token_target, token_source]
                    max_j = j
            if max_j > 0:
                writer.write(str(dk + 1) + ' ' + str(source_tokens[max_j]) + ' ' + str(target_tokens[i]) + '\n')
    writer.flush()
    writer.close()
    """


