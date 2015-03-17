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
import pdb


np.set_printoptions(precision=4, linewidth=180)


def write_probs(save_trans, translations, const_translations, source_to_target, check_sum=False):
    TYPE = "EMISSION"
    writer = open(save_trans, 'w')
    for k in sorted(translations):
        v = translations[k]
        if check_sum:
            t, s = k
            sum_to_1 = 0.0
            for t in source_to_target[s]:
                sum_to_1 += translations[t, s]
            assert abs(sum_to_1 - 1.0) < 1.0e-10

        writer.write(TYPE + '\t' + str('\t'.join(k)) + '\t' + str(v) + '\t\t\t\t' + str(
            const_translations[k]) + '\n')
    writer.flush()
    writer.close()


def make_const(init_trans):
    const_trans = {}
    for key in init_trans:
        r = np.random.randint(1, 3)
        if r == 2:
            const_trans[key] = float(np.random.randint(1, 15))
        else:
            const_trans[key] = 1.0
    return const_trans


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
    opt.add_option("-p", dest="save_trans", default="model1.probs.c")
    (options, _) = opt.parse_args()
    print '#', options
    source = options.source_corpus
    target = options.target_corpus
    ali_target = options.target_test if options.target_test is not None else options.target_corpus
    ali_source = options.source_test if options.source_test is not None else options.source_corpus
    save_trans = options.save_trans

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
            try:
                [emission, t, s, p, extra1, extra2] = line.split()
            except ValueError, err:
                [emission, t, s, p] = line.split()
            translations[t, s] = float(p)

    const_translations = make_const(translations)
    source_to_target = {}
    target_sentences = []
    source_sentences = []
    for s, t in zip(corpus_source, corpus_target):
        s = ['NULL'] + s.strip().split()
        t = t.strip().split()
        source_sentences.append(s)
        target_sentences.append(t)

    from cyth.cyth_model1 import model1_const

    translations, source_to_target = model1_const(5, translations, source_sentences, target_sentences,
                                                  const_translations, source_to_target)

    write_probs(save_trans, translations, const_translations, source_to_target, check_sum=True)



