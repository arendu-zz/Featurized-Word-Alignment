import numpy as np

def load_corpus_file(corpus_file):
    sentences = []
    types = set([])
    for l in open(corpus_file, 'r').readlines():
        sent = l.strip().split()
        sentences.append(sent)
        types.update(sent)
    return sentences, types

def get_source_to_target_firing(eve_to_feat):
    src_to_tar = {}
    for e in eve_to_feat:
        event_type, decision, context = e
        tar_l = src_to_tar.get(context, set([]))
        tar_l.add(decision)
        src_to_tar[context] = tar_l
    return src_to_tar

def load_model1_probs(path_model1_probs):
    m1_probs = {}
    for line in open(path_model1_probs, 'r').readlines():
        try:
            prob_type, tar, src, prob, ex1 = line.strip().split()
        except ValueError, err:
            prob_type, tar, src, prob = line.strip().split()
        prob = float(prob)
        m1_probs[tar, src] = prob
    m1_probs["#END#", "#END#"] = 1.0
    m1_probs["#START#", "#START#"] = 1.0
    return m1_probs

def pre_compute_ets(m1_probs, src_to_tar, t_types, s_types):
    _ets = {}
    c = 0
    for s in s_types:
        sum_s = 0.0
        t1 = t_types - src_to_tar.get(s, set([]))  # TODO: can speed up by only iterating over co-occuring t_types
        for t in t1:
            if t is "#START#" and s is "#START#":
                pass
            elif t is "#END#" and s is "#END#":
                pass
            elif (t, s) in m1_probs:
                c += 1
                sum_s += m1_probs[t, s]
            else:
                # t,s do not co-occur so we are adding 0.0 to sum_s (or -inf to log sum_s)
                pass
        _ets[s] = sum_s
    _ets["#START#"] = 1.0
    _ets["#END#"] = 1.0
    return _ets

def load_dictionary_features(dict_features_path=None):
    dictionary_features = {}
    if dict_features_path is None:
        print 'no dictionary features...'
        return dictionary_features
    else:
        df = open(dict_features_path, 'r').readlines()
        for line in df:
            line = line.strip()
            if line is not '':
                terms, v = line.strip().split('\t')
                t1, t2 = terms.split('|||')
                dictionary_features[t1, t2] = v
        print 'loaded ', len(dictionary_features), ' dictionary features...'
        return dictionary_features

def get_wa_features_fired(type, decision, context, dictionary_features, hybrid=False):
    fired_features = []
    if type == "EMISSION":
        if not hybrid:
            fired_features = [(1.0, ("EMISSION", decision, context))]

        if decision == context:
            fired_features += [(1.0, ("IS_SAME", decision, context))]

        if dictionary_features is not None and (decision, context) in dictionary_features:
            fired_features += [(1.0, ("IN_DICT", decision, context))]

        if decision[0:3] == context[0:3] and hybrid:
            fired_features += [(1.0, ("PREFFIX3", decision[0:3], context[0:3]))]

        if decision[-3:] == context[-3:] and hybrid:
            fired_features += [(1.0, ("SUFFIX3", decision[-3:], context[-3:]))]

        # if len(decision) == len(context) and hybrid:
        # fired_features += [(1.0, ("SAME_LEN", len(decision), len(context)))]

        # if context == "NULL":
        # fired_features += [(-1.0, ("IS_FROM_NULL", context))]

        # if const.has_pos:
        # decision_pos = decision.split("_")[1]
        # context_pos = context.split("_")[1]
        # if decision_pos == context_pos:
        # fired_features += [(1.0, ("IS_POS_SAME", decision_pos, context_pos))]

        """if decision[0].isupper() and context[0].isupper() and context != "NULL":
            fired_features += [("IS_UPPER", decision, context)]"""
    elif type == "TRANSITION":
        p = context
        if decision != "NULL" and p != "NULL":
            jump = abs(decision - p)
        else:
            jump = "NULL"
        fired_features = [(1.0, ("TRANSITION", jump))]

    return fired_features

def populate_trellis(source_corpus, target_corpus, max_jump_width, max_beam_width):
    trellis = []
    for s_sent, t_sent in zip(source_corpus, target_corpus):
        t_sent.insert(0, "#START#")
        t_sent.append("#END#")
        s_sent.insert(0, "#START#")
        s_sent.append("#END#")
        trelli = {}
        for t_idx, t_tok in enumerate(t_sent):
            if t_idx == 0:
                state_options = [(t_idx, s_sent.index("#START#"))]
            elif t_idx == len(t_sent) - 1:
                state_options = [(t_idx, s_sent.index("#END#"))]
            else:
                state_options = [(t_idx, s_idx) for s_idx, s_tok in enumerate(s_sent) if
                                 s_tok != "#END#" and s_tok != "#START#"]
            trelli[t_idx] = state_options
        """
        # print 'fwd prune'
        for t_idx in sorted(trelli.keys())[1:-1]:
            # print t_idx
            p_t_idx = t_idx - 1
            p_max_s_idx = max(trelli[p_t_idx])[1]
            p_min_s_idx = min(trelli[p_t_idx])[1]
            j_max_s_idx = p_max_s_idx + max_jump_width
            j_min_s_idx = p_min_s_idx - max_jump_width if p_min_s_idx - max_jump_width >= 1 else 1
            c_filtered = [(t, s) for t, s in trelli[t_idx] if (j_max_s_idx >= s >= j_min_s_idx)]
            trelli[t_idx] = c_filtered
        # print 'rev prune'
        for t_idx in sorted(trelli.keys(), reverse=True)[1:-1]:
            # print t_idx
            p_t_idx = t_idx + 1
            try:
                p_max_s_idx = max(trelli[p_t_idx])[1]
                p_min_s_idx = min(trelli[p_t_idx])[1]
            except ValueError:
                raise BaseException("Jump value too small to form trellis")
            # print 'max', 'min', p_max_s_idx, p_min_s_idx
            j_max_s_idx = p_max_s_idx + max_jump_width
            j_min_s_idx = p_min_s_idx - max_jump_width if p_min_s_idx - max_jump_width >= 1 else 1
            # print 'jmax', 'jmin', j_max_s_idx, j_min_s_idx
            c_filtered = [(t, s) for t, s in trelli[t_idx] if (j_max_s_idx >= s >= j_min_s_idx)]
            trelli[t_idx] = c_filtered

        # beam prune
        for t_idx in sorted(trelli.keys())[1:-1]:
            x = trelli[t_idx]
            y = sorted([(abs(t - s), (t, s)) for t, s in x])
            py = sorted([ts for d, ts in y[:max_beam_width]])
            trelli[t_idx] = py
        """
        for t_idx in sorted(trelli.keys())[1:-1]:
            trelli[t_idx] += [(t_idx, "NULL")]
        trellis.append(trelli)
    return trellis

def populate_features(trellis, source, target, model_type, dictionary_features):
    events_to_features = {}
    features_to_events = {}
    feature_index = {}
    feature_counts = {}
    event_index = set([])
    event_to_event_index = {}
    event_counts = {}
    normalizing_decision_map = {}

    for treli_idx, treli in enumerate(trellis):
        for idx in treli:
            for t_idx, s_idx in treli[idx]:
                t_tok = target[treli_idx][t_idx]
                if s_idx is "NULL":
                    s_tok = "NULL"
                else:
                    s_tok = source[treli_idx][s_idx]
                """
                emission features
                """
                ndm = normalizing_decision_map.get(("EMISSION", s_tok), set([]))
                ndm.add(t_tok)
                normalizing_decision_map["EMISSION", s_tok] = ndm
                emission_context = s_tok
                emission_decision = t_tok
                emission_event = ("EMISSION", emission_decision, emission_context)
                event_index.add(emission_event)
                event_counts[emission_event] = event_counts.get(emission_event, 1.0)
                ff_e = get_wa_features_fired(type="EMISSION", decision=emission_decision, context=emission_context,
                                             hybrid=model_type == "hybrid_model1",
                                             dictionary_features=dictionary_features)
                for f_wt, f in ff_e:
                    feature_index[f] = len(feature_index) if f not in feature_index else feature_index[f]
                    ca2f = events_to_features.get(emission_event, set([]))
                    ca2f.add(f)
                    events_to_features[emission_event] = ca2f
                    f2ca = features_to_events.get(f, set([]))
                    f2ca.add(emission_event)
                    features_to_events[f] = f2ca
                    feature_counts[f] = feature_counts.get(f, 0.0) + 1.0

                if idx > 0 and model_type == "hmm":
                    for prev_t_idx, prev_s_idx in treli[idx - 1]:
                        """
                        transition features
                        """
                        transition_context = prev_s_idx
                        transition_decision = s_idx
                        transition_event = ("TRANSITION", transition_decision, transition_context)
                        event_index.add(transition_event)
                        event_counts[transition_event] = event_counts.get(transition_event, 1.0)
                        ff_t = get_wa_features_fired(type="TRANSITION", decision=transition_decision,
                                                     context=transition_context,
                                                     dictionary_features=dictionary_features)

                        ndm = normalizing_decision_map.get(("TRANSITION", transition_context), set([]))
                        ndm.add(transition_decision)
                        normalizing_decision_map["TRANSITION", transition_context] = ndm
                        for f_wt, f in ff_t:
                            feature_index[f] = len(feature_index) if f not in feature_index else feature_index[f]
                            ca2f = events_to_features.get(transition_event, set([]))
                            ca2f.add(f)
                            events_to_features[transition_event] = ca2f
                            f2ca = features_to_events.get(f, set([]))
                            f2ca.add(transition_event)
                            features_to_events[f] = f2ca
                            feature_counts[f] = feature_counts.get(f, 0.0) + 1.0

    du = np.zeros(len(feature_index))
    for f in feature_index:
        i = feature_index[f]
        c = feature_counts[f]
        du[i] = c

    event_index = sorted(list(event_index))
    for ei, e in enumerate(event_index):
        event_to_event_index[e] = ei

    return events_to_features, \
           features_to_events, \
           feature_index, \
           feature_counts, \
           event_index, \
           event_to_event_index, \
           event_counts, \
           normalizing_decision_map, \
           du

def initialize_theta(input_weights_file, feature_index, rand=False):
    if rand:
        init_theta = np.random.uniform(-1.0, 1.0, len(feature_index))
    else:
        init_theta = np.random.uniform(1.0, 1.0, len(feature_index))
    if input_weights_file is not None:
        print 'reading initial weights...'
        for l in open(input_weights_file, 'r').readlines():
            l_key = tuple(l.split()[:-1])
            if l_key in feature_index:
                init_theta[feature_index[l_key]] = float(l.split()[-1:][0])
            else:
                pass
    else:
        print 'no initial weights given, random initial weights assigned...'
    return init_theta

def write_probs(theta, save_probs, fractional_counts, func):
    write_probs = open(save_probs, 'w')
    for fc in sorted(fractional_counts):
        (t, d, c) = fc
        if t == "EMISSION" or t == "TRANSITION":
            prob = func(theta, type=t, decision=d, context=c)
            str_t = reduce(lambda a, d: str(a) + '\t' + str(d), fc, '')
            write_probs.write(str_t.strip() + '\t' + str(round(prob, 5)) + '' + "\n")
    write_probs.flush()
    write_probs.close()
    print 'wrote probs to:', save_probs

def write_weights(theta, save_weights, feature_index):
    write_theta = open(save_weights, 'w')
    for t in sorted(feature_index):
        str_t = reduce(lambda a, d: str(a) + '\t' + str(d), t, '')
        write_theta.write(str_t.strip() + '\t' + str(theta[feature_index[t]]) + '' + "\n")
    write_theta.flush()
    write_theta.close()
    print 'wrote weights to:', save_weights

def write_alignments_col_tok(theta, save_align, trellis, source, target, func):
    save_align += '.col.tokens'
    write_align = open(save_align, 'w')
    # write_align.write(snippet)
    for idx, obs in enumerate(trellis[:]):
        max_bt = func(theta, idx)[0]
        for tar_i, src_i in max_bt:
            if src_i is not "NULL" and src_i > 0 and tar_i > 0:
                write_align.write(str(idx + 1) + ' ' + source[idx][src_i] + ' ' + target[idx][tar_i] + '\n')
    write_align.flush()
    write_align.close()
    print 'wrote alignments to:', save_align

def write_alignments_col(theta, save_align, trellis, func):
    save_align += '.col'

    write_align = open(save_align, 'w')
    # write_align.write(snippet)
    for idx, obs in enumerate(trellis[:]):
        max_bt = func(theta, idx)[0]
        for tar_i, src_i in max_bt:
            if src_i is not "NULL" and tar_i > 0 and src_i > 0:
                write_align.write(str(idx + 1) + ' ' + str(src_i) + ' ' + str(tar_i) + '\n')
    write_align.flush()
    write_align.close()
    print 'wrote alignments to:', save_align

def write_alignments(theta, save_align, trellis, func):
    write_align = open(save_align, 'w')
    # write_align.write(snippet)
    for idx, obs in enumerate(trellis[:]):
        max_bt = func(theta, idx)[0]
        w = ' '.join(
            [str(src_i) + '-' + str(tar_i) for tar_i, src_i in max_bt if
             src_i is not "NULL" and tar_i > 0 and src_i > 0])
        write_align.write(w + '\n')
    write_align.flush()
    write_align.close()
    print 'wrote alignments to:', save_align
