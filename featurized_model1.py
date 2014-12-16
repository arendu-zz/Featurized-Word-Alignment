__author__ = 'arenduchintala'

from optparse import OptionParser
from math import exp, log
import sys
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
import numpy as np
import FeatureEng as FE
import utils
import random
import copy
import pdb
from pprint import pprint

global BOUNDARY_START, END_STATE, SPLIT, E_TYPE, T_TYPE
global cache_normalizing_decision, features_to_events, events_to_features, normalizing_decision_map
global trellis, max_jump_width, number_of_events, EPS, snippet, max_beam_width, rc
global source, target, data_likelihood, event_grad
event_grad = {}
data_likelihood = 0.0
snippet = ''
EPS = 1e-5
rc = 0.25

max_jump_width = 1000
max_beam_width = 20  # creates a span of +/- span centered around current token
trellis = []
cache_normalizing_decision = {}
BOUNDARY_START = "#START#"
BOUNDARY_END = "#END#"
NULL = "NULL"
E_TYPE = "EMISSION"
E_TYPE_PRE = "PREFIX_FEATURE"
E_TYPE_SUF = "SUFFIX_FEATURE"
T_TYPE = "TRANSITION"
ALL = "ALL_STATES"
fractional_counts = {}
number_of_events = 0
events_to_features = {}
features_to_events = {}
feature_index = {}
conditional_arc_index = {}
normalizing_decision_map = {}
pause_on_tie = False


def populate_features():
    global trellis, feature_index, source, target
    for treli_idx, treli in enumerate(trellis):
        for idx in treli:
            for t_idx, s_idx in treli[idx]:
                t_tok = target[treli_idx][t_idx]
                if s_idx == NULL:
                    s_tok = NULL
                else:
                    s_tok = source[treli_idx][s_idx]
                """
                emission features
                """
                ndm = normalizing_decision_map.get((E_TYPE, s_tok), set([]))
                ndm.add(t_tok)
                normalizing_decision_map[E_TYPE, s_tok] = ndm
                emission_context = s_tok
                emission_decision = t_tok
                emission_event = (E_TYPE, emission_decision, emission_context)
                ff_e = FE.get_wa_features_fired(type=E_TYPE, decision=emission_decision, context=emission_context)
                for f_wt, f in ff_e:
                    feature_index[f] = len(feature_index) if f not in feature_index else feature_index[f]
                    ca2f = events_to_features.get(emission_event, set([]))
                    ca2f.add(f)
                    events_to_features[emission_event] = ca2f
                    f2ca = features_to_events.get(f, set([]))
                    f2ca.add(emission_event)
                    features_to_events[f] = f2ca


def get_decision_given_context(theta, type, decision, context):
    global normalizing_decision_map, cache_normalizing_decision, feature_index
    fired_features = FE.get_wa_features_fired(type=type, context=context, decision=decision)

    theta_dot_features = sum([theta[feature_index[f]] * f_wt for f_wt, f in fired_features])

    if (type, context) in cache_normalizing_decision:
        theta_dot_normalizing_features = cache_normalizing_decision[type, context]
    else:
        normalizing_decisions = normalizing_decision_map[type, context]
        theta_dot_normalizing_features = 0
        for d in normalizing_decisions:
            d_features = FE.get_wa_features_fired(type=type, context=context, decision=d)
            theta_dot_normalizing_features += exp(sum([theta[feature_index[f]] * f_wt for f_wt, f in d_features]))

        theta_dot_normalizing_features = log(theta_dot_normalizing_features)
        cache_normalizing_decision[type, context] = theta_dot_normalizing_features
    log_prob = round(theta_dot_features - theta_dot_normalizing_features, 10)
    if log_prob > 0.0:
        # print "log_prob = ", log_prob, type, decision, context
        # pdb.set_trace()
        if options.algorithm == 'LBFGS':
            raise Exception
        else:
            log_prob = 0.0  # TODO figure out why in the EM algorithm this error happens?
    return log_prob


def get_model1_forward(theta, obs_id):
    global fractional_counts, source, target, trellis
    obs = trellis[obs_id]
    max_bt = [-1] * len(obs)
    p_st = 0.0
    for t_idx in obs:
        t_tok = target[obs_id][t_idx]
        sum_e = float('-inf')
        max_e = float('-inf')
        max_s_idx = None
        sum_sj = float('-inf')
        for _, s_idx in obs[t_idx]:
            s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
            e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
            sum_e = utils.logadd(sum_e, e)
            q = log(1.0 / len(obs[t_idx]))
            sum_sj = utils.logadd(sum_sj, e + q)
            if e > max_e:
                max_e = e
                max_s_idx = s_idx
        max_bt[t_idx] = (t_idx, max_s_idx)
        p_st += sum_sj

        # update fractional counts
        for _, s_idx in obs[t_idx]:
            s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
            e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
            delta = e - sum_e
            event = (E_TYPE, t_tok, s_tok)
            fractional_counts[event] = utils.logadd(delta, fractional_counts.get(event, float('-inf')))

    return p_st, max_bt[:-1]


def reset_fractional_counts():
    global fractional_counts, cache_normalizing_decision, number_of_events
    fractional_counts = {}  # dict((k, float('-inf')) for k in conditional_arc_index)
    cache_normalizing_decision = {}
    number_of_events = 0


def write_probs(theta, save_probs):
    global feature_index
    write_probs = open(save_probs, 'w')
    write_probs.write(snippet)
    for fc in sorted(fractional_counts):
        (t, d, c) = fc
        prob = get_decision_given_context(theta, type=t, decision=d, context=c)
        str_t = reduce(lambda a, d: str(a) + '\t' + str(d), fc, '')
        write_probs.write(str_t.strip() + '\t' + str(round(prob, 5)) + '' + "\n")
    write_probs.flush()
    write_probs.close()
    print 'wrote probs to:', save_probs


def write_weights(theta, save_weights):
    global trellis, feature_index
    write_theta = open(save_weights, 'w')
    write_theta.write(snippet)
    for t in sorted(feature_index):
        str_t = reduce(lambda a, d: str(a) + '\t' + str(d), t, '')
        write_theta.write(str_t.strip() + '\t' + str(theta[feature_index[t]]) + '' + "\n")
    write_theta.flush()
    write_theta.close()
    print 'wrote weights to:', save_weights


def write_alignments_col_tok(theta, save_align):
    save_align += '.col.tokens'
    global trellis, feature_index, source, target
    write_align = open(save_align, 'w')
    # write_align.write(snippet)
    for idx, obs in enumerate(trellis[:]):
        S, max_bt = get_model1_forward(theta, idx)
        for tar_i, src_i in max_bt:
            if src_i != NULL and src_i > 0 and tar_i > 0:
                write_align.write(str(idx + 1) + ' ' + source[idx][src_i] + ' ' + target[idx][tar_i] + '\n')
    write_align.flush()
    write_align.close()
    print 'wrote alignments to:', save_align


def write_alignments_col(theta, save_align):
    save_align += '.col'
    global trellis, feature_index
    write_align = open(save_align, 'w')
    # write_align.write(snippet)
    for idx, obs in enumerate(trellis[:]):
        S, max_bt = get_model1_forward(theta, idx)
        for tar_i, src_i in max_bt:
            if src_i != NULL and tar_i > 0 and src_i > 0:
                write_align.write(str(idx + 1) + ' ' + str(src_i) + ' ' + str(tar_i) + '\n')
    write_align.flush()
    write_align.close()
    print 'wrote alignments to:', save_align


def write_alignments(theta, save_align):
    global trellis, feature_index
    write_align = open(save_align, 'w')
    # write_align.write(snippet)
    for idx, obs in enumerate(trellis[:]):
        S, max_bt = get_model1_forward(theta, idx)
        w = ' '.join(
            [str(src_i) + '-' + str(tar_i) for tar_i, src_i in max_bt if src_i != NULL and tar_i > 0 and src_i > 0])
        write_align.write(w + '\n')
    write_align.flush()
    write_align.close()
    print 'wrote alignments to:', save_align


def get_likelihood(theta):
    assert isinstance(theta, np.ndarray)
    assert len(theta) == len(feature_index)
    global trellis, data_likelihood, rc
    reset_fractional_counts()
    data_likelihood = 0.0

    batch = range(0, len(trellis))

    for idx in batch:
        S, max_bt = get_model1_forward(theta, idx)
        # print 'p(t|s) for', idx, ':', S, max_bt
        data_likelihood += S

    reg = np.sum(theta ** 2)
    ll = data_likelihood - (rc * reg)

    print 'log likelihood:', ll
    return -ll


def get_likelihood_with_expected_counts(theta):
    global fractional_counts
    sum_likelihood = 0.0
    for event in fractional_counts:
        (t, d, c) = event
        A_dct = exp(fractional_counts[event])
        a_dct = get_decision_given_context(theta=theta, type=t, decision=d, context=c)
        sum_likelihood += A_dct * a_dct
    reg = np.sum(theta ** 2)
    sum_likelihood -= (rc * reg)

    print '\tec log likelihood:', sum_likelihood
    return -sum_likelihood


def get_gradient(theta):
    global fractional_counts, feature_index, event_grad, rc
    assert len(theta) == len(feature_index)
    event_grad = {}
    for event_j in fractional_counts:
        (t, dj, cj) = event_j
        f_val, f = FE.get_wa_features_fired(type=t, context=cj, decision=dj)[0]
        a_dp_ct = exp(get_decision_given_context(theta, decision=dj, context=cj, type=t)) * f_val
        sum_feature_j = 0.0
        norm_events = [(t, dp, cj) for dp in normalizing_decision_map[t, cj]]
        for event_i in norm_events:
            A_dct = exp(fractional_counts.get(event_i, 0.0))
            if event_i == event_j:
                (ti, di, ci) = event_i
                fj, f = FE.get_wa_features_fired(type=ti, context=ci, decision=di)[0]
            else:
                fj = 0.0
            sum_feature_j += A_dct * (fj - a_dp_ct)
        event_grad[event_j] = sum_feature_j  # - abs(theta[event_j])  # this is the regularizing term

    # grad = np.zeros_like(theta)
    grad = -2 * rc * theta  # l2 regularization with lambda 0.5
    for e in event_grad:
        feats = events_to_features[e]
        for f in feats:
            grad[feature_index[f]] += event_grad[e]

    # for s in seen_index:
    # grad[s] += -theta[s]  # l2 regularization with lambda 0.5
    assert len(grad) == len(feature_index)
    return -grad


def populate_trellis(source_corpus, target_corpus):
    global max_jump_width, max_beam_width
    new_trellis = []
    for s_sent, t_sent in zip(source_corpus, target_corpus):
        t_sent.insert(0, BOUNDARY_START)
        t_sent.append(BOUNDARY_END)
        s_sent.insert(0, BOUNDARY_START)
        s_sent.append(BOUNDARY_END)
        trelli = {}
        for t_idx, t_tok in enumerate(t_sent):
            if t_idx == 0:
                state_options = [(t_idx, s_sent.index(BOUNDARY_START))]
            elif t_idx == len(t_sent) - 1:
                state_options = [(t_idx, s_sent.index(BOUNDARY_END))]
            else:
                state_options = [(t_idx, s_idx) for s_idx, s_tok in enumerate(s_sent) if
                                 s_tok != BOUNDARY_END and s_tok != BOUNDARY_START]
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
            trelli[t_idx] += [(t_idx, NULL)]
        new_trellis.append(trelli)
    return new_trellis


def gradient_check_em():
    global EPS, feature_index
    init_theta = initialize_theta(None)
    f_approx = {}
    for f in feature_index:
        theta_plus = copy.deepcopy(init_theta)
        theta_minus = copy.deepcopy(init_theta)
        theta_plus[feature_index[f]] = init_theta[feature_index[f]] + EPS
        get_likelihood(theta_plus)  # updates fractional counts
        val_plus = get_likelihood_with_expected_counts(theta_plus)
        theta_minus[feature_index[f]] = init_theta[feature_index[f]] - EPS
        get_likelihood(theta_minus)  # updates fractional counts
        val_minus = get_likelihood_with_expected_counts(theta_minus)
        f_approx[f] = (val_plus - val_minus) / (2 * EPS)

    my_grad = get_gradient(init_theta)
    diff = []
    for k in sorted(f_approx):
        diff.append(abs(my_grad[feature_index[k]] - f_approx[k]))
        print str(round(my_grad[feature_index[k]] - f_approx[k], 3)).center(10), str(
            round(my_grad[feature_index[k]], 5)).center(10), \
            str(round(f_approx[k], 5)).center(10), k
    f_approx = sorted([(feature_index[k], v) for k, v in f_approx.items()])
    f_approx = np.array([v for k, v in f_approx])

    print 'component difference:', round(sum(diff), 3), \
        'cosine similarity:', utils.cosine_sim(f_approx, my_grad), \
        ' sign difference', utils.sign_difference(f_approx, my_grad)


def gradient_check_lbfgs():
    global EPS, feature_index
    init_theta = initialize_theta(None)
    chk_grad = utils.gradient_checking(init_theta, EPS, get_likelihood)
    my_grad = get_gradient(init_theta)
    diff = []
    for f in sorted(feature_index):  # xrange(len(chk_grad)):
        k = feature_index[f]
        diff.append(abs(my_grad[k] - chk_grad[k]))
        print str(round(my_grad[k] - chk_grad[k], 5)).center(10), str(
            round(my_grad[k], 5)).center(10), \
            str(round(chk_grad[k], 5)).center(10), f

    print 'component difference:', round(sum(diff), 3), \
        'cosine similarity:', utils.cosine_sim(chk_grad, my_grad), \
        ' sign difference', utils.sign_difference(chk_grad, my_grad)


def initialize_theta(input_weights):
    global feature_index
    init_theta = np.random.uniform(1.0, 1.0, len(feature_index))
    if input_weights is not None:
        print 'reading initial weights...'
        for l in open(options.input_weights, 'r').readlines():
            l_key = tuple(l.split()[:-1])
            if l_key in feature_index:
                init_theta[feature_index[l_key]] = float(l.split()[-1:][0])
                # print 'updated ', l_key
            else:
                # print 'ignored', l_key
                pass
    else:
        print 'no initial weights given, random initial weights assigned...'
    return init_theta


def write_logs(theta, current_iter):
    global trellis
    feature_val_typ = 'bin' if options.feature_values is None else 'real'
    name_prefix = '.'.join(
        [options.algorithm, str(rc), 'simple-model1', feature_val_typ])
    if current_iter is not None:
        name_prefix += '.' + str(current_iter)
    write_weights(theta, name_prefix + '.' + options.output_weights)
    write_probs(theta, name_prefix + '.' + options.output_probs)

    if options.source_test is not None and options.target_test is not None:
        source = [s.strip().split() for s in open(options.source_test, 'r').readlines()]
        target = [t.strip().split() for t in open(options.target_test, 'r').readlines()]
        trellis = populate_trellis(source, target)

    write_alignments(theta, name_prefix + '.' + options.output_alignments)
    write_alignments_col(theta, name_prefix + '.' + options.output_alignments)
    write_alignments_col_tok(theta, name_prefix + '.' + options.output_alignments)


if __name__ == "__main__":
    trellis = []

    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/toy.fr")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/toy.en")
    opt.add_option("--tt", dest="target_test", default="experiment/data/toy.fr")
    opt.add_option("--ts", dest="source_test", default="experiment/data/toy.en")

    opt.add_option("--iw", dest="input_weights", default=None)
    opt.add_option("--fv", dest="feature_values", default=None)
    opt.add_option("--ow", dest="output_weights", default="theta", help="extention of trained weights file")
    opt.add_option("--oa", dest="output_alignments", default="alignments", help="extension of alignments files")
    opt.add_option("--op", dest="output_probs", default="probs", help="extension of probabilities")
    opt.add_option("-g", dest="test_gradient", default="false")
    opt.add_option("-r", dest="regularization_coeff", default="0.0")
    opt.add_option("-a", dest="algorithm", default="LBFGS",
                   help="use 'EM' 'LBFGS' 'SGD'")

    (options, _) = opt.parse_args()
    rc = float(options.regularization_coeff)
    source = [s.strip().split() for s in open(options.source_corpus, 'r').readlines()]
    target = [s.strip().split() for s in open(options.target_corpus, 'r').readlines()]
    trellis = populate_trellis(source, target)
    populate_features()
    FE.load_feature_values(options.feature_values)
    snippet = "#" + str(opt.values) + "\n"

    if options.algorithm == "LBFGS":
        if options.test_gradient.lower() == "true":
            gradient_check_lbfgs()
        else:
            print 'skipping gradient check...'
            init_theta = initialize_theta(options.input_weights)
            t1 = minimize(get_likelihood, init_theta, method='L-BFGS-B', jac=get_gradient, tol=1e-2,
                          options={'maxiter': 20})

            theta = t1.x

    elif options.algorithm == "EM":
        if options.test_gradient.lower() == "true":
            gradient_check_em()
        else:
            print 'skipping gradient check...'
            theta = initialize_theta(options.input_weights)
            new_e = get_likelihood(theta)
            exp_new_e = get_likelihood_with_expected_counts(theta)
            old_e = float('-inf')
            converged = False
            iterations = 0
            while not converged and iterations < 5:
                t1 = minimize(get_likelihood_with_expected_counts, theta, method='L-BFGS-B', jac=get_gradient, tol=1e-2,
                              options={'maxiter': 20})
                theta = t1.x
                new_e = get_likelihood(theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 1) == 0.0
                old_e = new_e
                iterations += 1

    if options.test_gradient.lower() == "true":
        pass
    else:
        write_logs(theta, current_iter=None)
