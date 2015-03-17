__author__ = 'arenduchintala'

from optparse import OptionParser
from math import exp, log
from scipy.optimize import minimize
import numpy as np

import utils
import copy

from const import NULL, BOUNDARY_START, BOUNDARY_END, HYBRID_MODEL_1, E_TYPE
from cyth.cyth_common import populate_trellis, populate_features, write_alignments, write_alignments_col, \
    write_alignments_col_tok, write_probs, write_weights, initialize_theta, get_wa_features_fired, \
    load_dictionary_features


global cache_normalizing_decision, features_to_events, events_to_features, normalizing_decision_map
global trellis, max_jump_width, number_of_events, EPS, snippet, max_beam_width, rc
global source, target, data_likelihood, event_grad, feature_index, event_index
global events_per_trellis, event_to_event_index, has_pos, event_counts, du, itercount, N, source_to_target_firing
global target_types, source_types, ets, features_and_context_to_decisions, dictionary_features
dictionary_features = {}
ets = {}
target_types = set([])
sources_types = set([])
has_pos = False
event_grad = {}
data_likelihood = 0.0
snippet = ''
EPS = 1e-5
rc = 0.25

max_jump_width = 10
max_beam_width = 20  # creates a span of +/- span centered around current token
trellis = []
cache_normalizing_decision = {}

fractional_counts = {}
number_of_events = 0
events_to_features = {}
features_to_events = {}
feature_index = {}
feature_counts = {}
features_and_context_to_decisions = {}
du = []
event_index = []
event_to_event_index = {}
event_counts = {}

normalizing_decision_map = {}
itercount = 0
pause_on_tie = False


def get_decision_given_context(theta, type, decision, context):
    global cache_normalizing_decision, feature_index, source_to_target_firing, model1_probs, ets
    m1_event_prob = model1_probs.get((decision, context), 0.0)
    fired_features = get_wa_features_fired(type=type, decision=decision, context=context,
                                           dictionary_features=dictionary_features, hybrid=True)
    theta_dot_features = sum([theta[feature_index[f]] * f_wt for f_wt, f in fired_features])
    numerator = m1_event_prob * exp(theta_dot_features)
    if (type, context) in cache_normalizing_decision:
        denom = cache_normalizing_decision[type, context]
    else:
        denom = ets[context]
        target_firings = source_to_target_firing.get(context, set([]))
        for tf in target_firings:
            m1_tf_event_prob = model1_probs.get((tf, context), 0.0)
            tf_fired_features = get_wa_features_fired(type=type, decision=tf, context=context,
                                                      dictionary_features=dictionary_features, hybrid=True)
            tf_theta_dot_features = sum([theta[feature_index[f]] * f_wt for f_wt, f in tf_fired_features])
            denom += m1_tf_event_prob * exp(tf_theta_dot_features)
        cache_normalizing_decision[type, context] = denom
    try:
        log_prob = log(numerator) - log(denom)
    except ValueError:
        print numerator, denom, decision, context, m1_event_prob, theta_dot_features
        raise BaseException
    return log_prob


def get_best_seq(theta, obs_id):
    global source, target, trellis
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

    return max_bt[:-1], p_st


def get_model1_forward(theta, obs_id, fc):
    global source, target, trellis
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
        if fc is not None:
            for _, s_idx in obs[t_idx]:
                s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
                e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
                delta = e - sum_e
                event = (E_TYPE, t_tok, s_tok)
                fc[event] = utils.logadd(delta, fc.get(event, float('-inf')))

    return max_bt[:-1], p_st, fc


def reset_fractional_counts():
    global fractional_counts, cache_normalizing_decision, number_of_events
    fractional_counts = {}  # dict((k, float('-inf')) for k in conditional_arc_index)
    cache_normalizing_decision = {}
    number_of_events = 0


def get_likelihood(theta):
    assert isinstance(theta, np.ndarray)
    assert len(theta) == len(feature_index)
    global trellis, data_likelihood, rc, fractional_counts
    reset_fractional_counts()
    data_likelihood = 0.0
    batch = range(0, len(trellis))
    for idx in batch:
        max_bt, S, fractional_counts = get_model1_forward(theta, idx, fractional_counts)
        # print 'p(t|s) for', idx, ':', S, max_bt
        data_likelihood += S

    reg = np.sum(theta ** 2)
    ll = data_likelihood - (rc * reg)
    e1 = get_decision_given_context(theta, E_TYPE, decision='.', context=NULL)
    e2 = get_decision_given_context(theta, E_TYPE, decision='.', context='.')
    print 'log likelihood:', ll, 'p(.|NULL)', e1, 'p(.|.)', e2
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
    for event_j in events_to_features:
        (t, dj, cj) = event_j
        f_val, f = \
            get_wa_features_fired(type=t, context=cj, decision=dj, dictionary_features=dictionary_features,
                                  hybrid=True)[0]
        a_dp_ct = exp(get_decision_given_context(theta, decision=dj, context=cj, type=t)) * f_val
        sum_feature_j = 0.0
        norm_events = [(t, dp, cj) for dp in normalizing_decision_map[t, cj]]
        for event_i in norm_events:
            A_dct = exp(fractional_counts.get(event_i, 0.0))
            if event_i == event_j:
                (ti, di, ci) = event_i
                fj, f = get_wa_features_fired(type=ti, context=ci, decision=di, dictionary_features=dictionary_features,
                                              hybrid=True)[0]
            else:
                fj = 0.0
            sum_feature_j += A_dct * (fj - a_dp_ct)
        event_grad[event_j] = sum_feature_j  # - abs(theta[event_j])  # this is the regularizing term


    # grad = np.zeros_like(theta)
    grad = -2 * rc * theta  # l2 regularization with lambda 0.5
    for e in event_grad:
        feats = events_to_features.get(e, [])
        for f in feats:
            grad[feature_index[f]] += event_grad[e]

    # for s in seen_index:
    # grad[s] += -theta[s]  # l2 regularization with lambda 0.5
    assert len(grad) == len(feature_index)
    return -grad


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


def gradient_check_lbfgs(init_theta):
    global EPS, feature_index
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


def batch_sgd_accumilate(obs_id):
    print obs_id


def load_model1_probs(path_model1_probs):
    m1_probs = {}
    for line in open(path_model1_probs, 'r').readlines():
        try:
            prob_type, tar, src, prob, ex1 = line.strip().split()
        except ValueError, err:
            prob_type, tar, src, prob = line.strip().split()
        prob = float(prob)
        m1_probs[tar, src] = prob
    m1_probs[BOUNDARY_END, BOUNDARY_END] = 1.0
    m1_probs[BOUNDARY_START, BOUNDARY_START] = 1.0
    return m1_probs


def pre_compute_ets(m1_probs, src_to_tar, t_types, s_types):
    _ets = {}
    c = 0
    for s in s_types:
        sum_s = 0.0
        t1 = t_types - src_to_tar.get(s, set([]))  # TODO: can speed up by only iterating over co-occuring t_types
        for t in t1:
            if t is BOUNDARY_START and s is BOUNDARY_START:
                pass
            elif t is BOUNDARY_END and s is BOUNDARY_END:
                pass
            elif (t, s) in m1_probs:
                c += 1
                sum_s += m1_probs[t, s]
            else:
                # t,s do not co-occur so we are adding 0.0 to sum_s (or -inf to log sum_s)
                pass
        _ets[s] = sum_s
    _ets[BOUNDARY_START] = 1.0
    _ets[BOUNDARY_END] = 1.0
    return _ets


def get_source_to_target_firing(eve_to_feat):
    src_to_tar = {}
    for e in eve_to_feat:
        event_type, decision, context = e
        tar_l = src_to_tar.get(context, set([]))
        tar_l.add(decision)
        src_to_tar[context] = tar_l
    return src_to_tar


def write_logs(theta, current_iter):
    global max_beam_width, max_jump_width, trellis, feature_index, fractional_counts
    feature_val_typ = 'bin' if options.feature_values is None else 'real'
    name_prefix = '.'.join(
        ['sp', options.algorithm, str(rc), 'hybrid-model1', feature_val_typ])
    if current_iter is not None:
        name_prefix += '.' + str(current_iter)
    write_weights(theta, name_prefix + '.' + options.output_weights, feature_index)
    write_probs(theta, name_prefix + '.' + options.output_probs, fractional_counts, get_decision_given_context)

    if options.source_test is not None and options.target_test is not None:
        source_test = [s.strip().split() for s in open(options.source_test, 'r').readlines()]
        target_test = [t.strip().split() for t in open(options.target_test, 'r').readlines()]
        trellis = populate_trellis(source_test, target_test, max_jump_width, max_beam_width)

    write_alignments(theta, name_prefix + '.' + options.output_alignments, trellis, get_best_seq)
    write_alignments_col(theta, name_prefix + '.' + options.output_alignments, trellis, get_best_seq)
    write_alignments_col_tok(theta, name_prefix + '.' + options.output_alignments, trellis, source_test, target_test,
                             get_best_seq)


if __name__ == "__main__":
    trellis = []

    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/dev.es")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/dev.en")
    opt.add_option("--tt", dest="target_test", default="experiment/data/dev.es")
    opt.add_option("--ts", dest="source_test", default="experiment/data/dev.en")
    opt.add_option("--df", dest="dict_features", default=None)
    opt.add_option("--m1", dest="model1_probs", default="experiment/data/model1.probs")
    opt.add_option("--iw", dest="input_weights", default=None)
    opt.add_option("--fv", dest="feature_values", default=None)
    opt.add_option("--ow", dest="output_weights", default="theta", help="extention of trained weights file")
    opt.add_option("--oa", dest="output_alignments", default="alignments", help="extension of alignments files")
    opt.add_option("--op", dest="output_probs", default="probs", help="extension of probabilities")
    opt.add_option("-g", dest="test_gradient", action="store_true", default=False)
    opt.add_option("-r", dest="regularization_coeff", default="0.0")
    opt.add_option("-a", dest="algorithm", default="LBFGS",
                   help="use 'EM' 'LBFGS' 'SGD'")

    (options, _) = opt.parse_args()
    rc = float(options.regularization_coeff)
    source = [s.strip().split() for s in open(options.source_corpus, 'r').readlines()]
    target = [s.strip().split() for s in open(options.target_corpus, 'r').readlines()]
    target_types = set(open(options.target_corpus, 'r').read().split())
    source_types = set(open(options.source_corpus, 'r').read().split())
    source_types.add(NULL)
    trellis = populate_trellis(source, target, max_jump_width, max_beam_width)

    dictionary_features = load_dictionary_features(options.dict_features)
    model1_probs = load_model1_probs(options.model1_probs)
    events_to_features, features_to_events, feature_index, feature_counts, event_index, event_to_event_index, event_counts, normalizing_decision_map, du = populate_features(
        trellis, source, target, HYBRID_MODEL_1, dictionary_features=dictionary_features)
    source_to_target_firing = get_source_to_target_firing(events_to_features)
    ets = pre_compute_ets(model1_probs, source_to_target_firing, target_types, source_types)
    print len(feature_index), 'features used...'
    snippet = "#" + str(opt.values) + "\n"
    if options.algorithm == "LBFGS":
        if options.test_gradient:
            init_theta = initialize_theta(options.input_weights, feature_index, rand=True)
            gradient_check_lbfgs()
        else:
            print 'skipping gradient check...'
            theta = initialize_theta(options.input_weights, feature_index)
            t1 = minimize(get_likelihood, theta, method='L-BFGS-B', jac=get_gradient, tol=1e-3,
                          options={'maxiter': 20})

            theta = t1.x

    elif options.algorithm == "EM":
        if options.test_gradient:
            gradient_check_em()
        else:
            theta = initialize_theta(options.input_weights, feature_index)
            print 'skipping gradient check...'
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

    if options.test_gradient == "true":
        pass
    else:
        write_logs(theta, current_iter=None)
