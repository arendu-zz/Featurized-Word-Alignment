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

np.seterr(all='raise')

from const import NULL, BOUNDARY_START, BOUNDARY_END, IBM_MODEL_1, HMM_MODEL, E_TYPE, T_TYPE, EPS, Po, LAMBDA_FEATURE, \
    use_lambda_feature
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


def sl(g1, r, l):
    return g1 * (1 - np.power(r, l)) / (1 - r)


def get_second_term(lf, obs_id, m, n, ):
    """
    global source, target, trellis
    obs = trellis[obs_id]
    m = len(obs)
    n =
    j_up = i * n / m
    j_down = j_up + 1
    """
    pass


def tl(g1, a1, r, d, n):
    g_np1 = g1 * np.power(r, n)
    a_n = d * (n - 1) + a1
    x_1 = a1 * g1
    g_2 = g1 * r
    # rm1 = r - 1
    # return (a_n * g_np1 - x_1) / rm1 - d * (g_np1 - g_2) / (rm1 * rm1)
    r1 = 1 - r
    return ((a_n * g_np1 - x_1) / r1) + (d * (g_np1 - g_2) / (r1 * r1))


def h(i, j, m, n):
    return - abs(i / m - j / n)


def z_lf(lf, i, m, n):
    j_up = i * n / m
    j_down = j_up + 1
    r = exp(-lf / n)
    el1 = exp(lf * h(i, j_up, m, n))
    el2 = exp(lf * h(i, j_down, m, n))
    sj_up = sl(el1, r, j_up)
    sj_down = sl(el2, r, n - j_down)
    s = sj_up + sj_down
    return s


def get_fast_align_transition(theta, i, j, m, n):
    if j == NULL:
        return np.log(Po)
    else:
        i, j, m, n = float(i), float(j), float(m), float(n)
        if use_lambda_feature:
            lf = theta[feature_index[LAMBDA_FEATURE]]

        """
        print i, j, m, n
        print 'h', h(i, j, m, n)
        print 'z', z_lf(lf, i, m, n)
        print 'e^lh/Z', (lf * h(i, j, m, n)) - np.log(z_lf(lf, i, m, n))
        print '*'
        """
        num = (lf * h(i, j, m, n))
        denum = z_lf(lf, i, m, n)
        try:
            t = np.log(1 - Po) + num - np.log(denum)
        except FloatingPointError:
            t = np.log(1 - Po)
        return t


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


def get_model1_forward(theta, obs_id, fc=None):
    global source, target, trellis
    obs = trellis[obs_id]
    m = len(obs)
    max_bt = [-1] * len(obs)
    p_st = 0.0
    for t_idx in obs:
        t_tok = target[obs_id][t_idx]
        sum_e = float('-inf')
        sum_pei = float('-inf')
        max_e = float('-inf')
        max_s_idx = None
        sum_sj = float('-inf')
        for _, s_idx in obs[t_idx]:
            n = len(obs[t_idx])
            s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
            e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
            sum_e = utils.logadd(sum_e, e)

            if t_tok == BOUNDARY_START or t_tok == BOUNDARY_END:
                q = 0.0
            else:
                q = get_fast_align_transition(theta, t_idx, s_idx, m - 2, n - 1)

            sum_pei = utils.logadd(sum_pei, q + e)
            sum_sj = utils.logadd(sum_sj, e + q)

            if e > max_e:
                max_e = e
                max_s_idx = s_idx
        max_bt[t_idx] = (t_idx, max_s_idx)
        p_st += sum_sj

        if p_st == float('inf'):
            pdb.set_trace()
        # update fractional counts
        if fc is not None:
            for _, s_idx in obs[t_idx]:
                s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
                e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)

                if t_tok == BOUNDARY_START or t_tok == BOUNDARY_END:
                    q = 0.0
                else:
                    q = get_fast_align_transition(theta, t_idx, s_idx, m - 2, n - 1)
                p_ai = e + q - sum_pei
                event = (E_TYPE, t_tok, s_tok)
                fc[event] = utils.logadd(p_ai, fc.get(event, float('-inf')))
                lambda_event1 = (LAMBDA_FEATURE, 'E_pai', None)
                lambda_event2 = (LAMBDA_FEATURE, 'E_delta', None)
                fc[lambda_event1] = utils.logadd(p_ai, fc.get(lambda_event1, float('-inf')))
                fc[lambda_event2] = utils.logadd(q, fc.get(lambda_event2, float('-inf')))

    return max_bt[:-1], p_st, fc


def reset_fractional_counts():
    global fractional_counts, cache_normalizing_decision, number_of_events
    fractional_counts = {}  # dict((k, float('-inf')) for k in conditional_arc_index)
    cache_normalizing_decision = {}
    number_of_events = 0


def get_lf_grad(theta):
    assert isinstance(theta, np.ndarray)
    assert len(theta) == len(feature_index)
    global trellis, data_likelihood, rc, fractional_counts, feature_index
    batch = range(0, len(trellis))
    for idx in batch:
        max_bt, S, fractional_counts = get_model1_forward(theta, idx, fractional_counts)
    lf_grad = -2 * rc * theta[feature_index[LAMBDA_FEATURE]]
    lambda_event1 = (LAMBDA_FEATURE, 'E_pai', None)
    lambda_event2 = (LAMBDA_FEATURE, 'E_delta', None)
    fc_e1 = fractional_counts[lambda_event1]
    fc_e2 = fractional_counts[lambda_event2]
    n = exp(fc_e1) - exp(fc_e2)
    lf_grad += n
    return lf_grad


def get_likelihood(theta):
    assert isinstance(theta, np.ndarray)
    assert len(theta) == len(feature_index)
    global trellis, data_likelihood, rc, fractional_counts, feature_index
    reset_fractional_counts()
    data_likelihood = 0.0
    batch = range(0, len(trellis))

    for idx in batch:
        max_bt, S, fractional_counts = get_model1_forward(theta, idx, fractional_counts)
        # print 'p(t|s) for', idx, ':', S  # , max_bt
        data_likelihood += S

    reg = np.sum(theta ** 2)
    ll = data_likelihood - (rc * reg)

    e1 = get_decision_given_context(theta, E_TYPE, decision='.', context=NULL)
    e2 = get_decision_given_context(theta, E_TYPE, decision='.', context='.')
    print 'log likelihood:', ll, 'p(.|NULL)', e1, 'p(.|.)', e2, 'lf', theta[feature_index[LAMBDA_FEATURE]]
    return -ll


def get_likelihood_with_expected_counts(theta):
    global fractional_counts
    sum_likelihood = 0.0
    for event in fractional_counts:
        (t, d, c) = event
        if t == E_TYPE:
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
        if t == E_TYPE:
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
    grad[feature_index[LAMBDA_FEATURE]] = 0
    for e in event_grad:
        feats = events_to_features[e]
        for f in feats:
            grad[feature_index[f]] += event_grad[e]

    # for s in seen_index:
    # grad[s] += -theta[s]  # l2 regularization with lambda 0.5
    """
    lambda_event1 = (LAMBDA_FEATURE, 'E_pai', None)
    lambda_event2 = (LAMBDA_FEATURE, 'E_delta', None)
    n = exp(fractional_counts[lambda_event1]) - exp(fractional_counts[lambda_event2])
    grad[feature_index[LAMBDA_FEATURE]] += -n
    """
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
    init_theta = initialize_theta(None, feature_index)
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

    write_alignments(theta, name_prefix + '.' + options.output_alignments, trellis, get_model1_forward)
    write_alignments_col(theta, name_prefix + '.' + options.output_alignments, trellis, get_model1_forward)
    write_alignments_col_tok(theta, name_prefix + '.' + options.output_alignments, trellis, source_test, target_test,
                             get_model1_forward)


if __name__ == "__main__":
    trellis = []

    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/train.es")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/train.en")
    opt.add_option("--tt", dest="target_test", default="experiment/data/train.es")
    opt.add_option("--ts", dest="source_test", default="experiment/data/train.en")

    opt.add_option("--iw", dest="input_weights", default=None)
    opt.add_option("--fv", dest="feature_values", default=None)
    opt.add_option("--ow", dest="output_weights", default="theta", help="extention of trained weights file")
    opt.add_option("--oa", dest="output_alignments", default="alignments", help="extension of alignments files")
    opt.add_option("--op", dest="output_probs", default="probs", help="extension of probabilities")
    opt.add_option("-g", dest="test_gradient", default="false")
    opt.add_option("-r", dest="regularization_coeff", default="0.0")
    opt.add_option("-a", dest="algorithm", default="EM",
                   help="use 'EM' 'LBFGS' 'SGD'")

    (options, _) = opt.parse_args()
    rc = float(options.regularization_coeff)

    source = [s.strip().split() for s in open(options.source_corpus, 'r').readlines()]
    target = [s.strip().split() for s in open(options.target_corpus, 'r').readlines()]
    trellis = populate_trellis(source, target, max_jump_width, max_beam_width)
    events_to_features, features_to_events, feature_index, feature_counts, event_index, event_to_event_index, event_counts, normalizing_decision_map = populate_features(
        trellis, source, target, IBM_MODEL_1)
    FE.load_feature_values(options.feature_values)
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
                              options={'maxiter': 10})
                theta = t1.x
                lf_grad = get_lf_grad(theta)
                print 'lf_grad', lf_grad
                theta[feature_index[LAMBDA_FEATURE]] += 0.01 * -lf_grad

                new_e = get_likelihood(theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 1) == 0.0
                old_e = new_e
                iterations += 1

    if options.test_gradient.lower() == "true":
        pass
    else:
        write_logs(theta, current_iter=None)
