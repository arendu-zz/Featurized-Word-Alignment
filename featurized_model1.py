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

from const import NULL, BOUNDARY_START, IBM_MODEL_1, HMM_MODEL, E_TYPE, T_TYPE, EPS
from common import populate_trellis, populate_features, write_alignments, write_alignments_col, \
    write_alignments_col_tok, write_probs, write_weights, initialize_theta


global cache_normalizing_decision, features_to_events, events_to_features, normalizing_decision_map
global trellis, max_jump_width, number_of_events, EPS, snippet, max_beam_width, rc
global source, target, data_likelihood, event_grad, feature_index, event_index
global events_per_trellis, event_to_event_index, has_pos, event_counts, du, itercount, N
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
du = []
event_index = []
event_to_event_index = {}
event_counts = {}

normalizing_decision_map = {}
itercount = 0
pause_on_tie = False


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


def batch_sgd_accumilate(obs_id):
    print obs_id


def write_logs(theta, current_iter):
    global max_beam_width, max_jump_width, trellis, feature_index, fractional_counts
    feature_val_typ = 'bin' if options.feature_values is None else 'real'
    name_prefix = '.'.join(
        ['sp', options.algorithm, str(rc), 'simple-model1', feature_val_typ])
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
    opt.add_option("-t", dest="target_corpus", default="experiment/data/dev.small.es")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/dev.small.en")
    opt.add_option("--tt", dest="target_test", default="experiment/data/dev.small.es")
    opt.add_option("--ts", dest="source_test", default="experiment/data/dev.small.en")
    opt.add_option("--df", dest="dict_features", default=None)
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
    trellis = populate_trellis(source, target, max_jump_width, max_beam_width)
    FE.load_feature_values(options.feature_values)
    FE.load_dictionary_features(options.dict_features)
    events_to_features, features_to_events, feature_index, feature_counts, event_index, event_to_event_index, event_counts, normalizing_decision_map, du = populate_features(
        trellis, source, target, IBM_MODEL_1)
    snippet = "#" + str(opt.values) + "\n"

    if options.algorithm == "LBFGS":
        if options.test_gradient.lower() == "true":
            gradient_check_lbfgs()
        else:
            print 'skipping gradient check...'
            init_theta = initialize_theta(options.input_weights, feature_index)
            t1 = minimize(get_likelihood, init_theta, method='L-BFGS-B', jac=get_gradient, tol=1e-3,
                          options={'maxiter': 20})

            theta = t1.x

    elif options.algorithm == "EM":
        if options.test_gradient.lower() == "true":
            gradient_check_em()
        else:
            print 'skipping gradient check...'
            theta = initialize_theta(options.input_weights, feature_index)
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
