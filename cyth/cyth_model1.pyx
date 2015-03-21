import numpy as np

def init_lex(source_sentences, target_sentences):
    co_occurance = {}
    translations = {}
    for k, (source_tokens, target_tokens) in enumerate(zip(source_sentences, target_sentences)):
        assert source_tokens[0] == 'NULL'
        for e in source_tokens:
            n_e = co_occurance.get(e, set())
            n_e.update(target_tokens)
            co_occurance[e] = n_e

    for k, val in co_occurance.iteritems():
        for v_es in val:
            translations[v_es, k] = 1.0 / len(val)
    return translations

def model1(iterations, translations, source_sentences, target_sentences):
    """
    :param iterations: number of iterations
    :param translations:  initial lexical translation probabilities
    :param source_sentences: a list of lists of source sentences (inner list is split() into words)
    :param target_sentences: a list of lists of target sentences (inner list is split() into words)
    :return: translations

    """
    for i in range(iterations):
        print 'iter',i
        delta = {}
        counts = {}
        """
        accumilate fractional counts, E-Step
        """
        for k, (source_tokens, target_tokens) in enumerate(zip(source_sentences, target_sentences)):
            if k % 1000 == 0:
                print k
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
    return translations

def model1_const(iterations, translations, source_sentences, target_sentences, const_translations,
                 source_to_target=None):
    """
    :param iterations: number of iterations
    :param translations:  initial lexical translation probabilities
    :param source_sentences: a list of lists of source sentences (inner list is split() into words)
    :param target_sentences: a list of lists of target sentences (inner list is split() into words)
    :param const_translations:  A dictionary which has the const scaling term for a specific (target_token,source_token) pair
    :param source_to_target: (optional) An empty dictionary which gets loaded with source-> list of targets it co-occurs with
    :return: translations, source_to_target
    """
    for iter in range(iterations):
        delta = {}
        counts = {}
        """
        accumilate fractional counts, E-Step
        """
        for k, (target_tokens, source_tokens) in enumerate(zip(target_sentences, source_sentences)):
            t_mat = np.zeros((len(target_tokens), len(source_tokens)))
            for j in range(0, len(source_tokens)):
                for i in range(0, len(target_tokens)):
                    t_mat[i][j] = translations[target_tokens[i], source_tokens[j]] * const_translations[
                        target_tokens[i], source_tokens[j]]
            t_sum = np.sum(t_mat, 1)

            for j in range(0, len(source_tokens)):
                for i in range(0, len(target_tokens)):
                    delta[k, i, j] = t_mat[i][j] / t_sum[i]
                    ck = (target_tokens[i], source_tokens[j])
                    sck = source_tokens[j]

                    counts[ck] = counts.get(ck, 0.0) + (delta[k, i, j] / const_translations[ck])
                    counts[sck] = counts.get(sck, 0.0) + (delta[k, i, j] / const_translations[ck])
                    if iter == 0 and source_to_target is not None:
                        s2t = source_to_target.get(source_tokens[j], set([]))
                        s2t.add(target_tokens[i])
                        source_to_target[source_tokens[j]] = s2t
                    else:
                        pass  # does not need to be done again

        """
        update translations, M-Step
        """
        for target_i, source_j in translations:
            translations[target_i, source_j] = counts[target_i, source_j] / counts[source_j]

    if source_to_target is not None:
        return translations, source_to_target
    else:
        return translations
