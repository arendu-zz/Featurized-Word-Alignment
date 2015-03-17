__author__ = 'arenduchintala'

from optparse import OptionParser
from math import exp, log
import sys
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
import numpy as np

import utils
import random
import copy
import pdb
from pprint import pprint
from const import NULL, BOUNDARY_START, IBM_MODEL_1, HMM_MODEL, E_TYPE, T_TYPE, EPS
from cyth.cyth_common import populate_trellis, populate_features, write_alignments, write_alignments_col, \
    write_alignments_col_tok, write_probs, write_weights, initialize_theta, load_dictionary_features, \
    get_wa_features_fired, load_corpus_file

global BOUNDARY_START, END_STATE, SPLIT, E_TYPE, T_TYPE, IBM_MODEL_1, HMM_MODEL
global cache_normalizing_decision, features_to_events, events_to_features, normalizing_decision_map
global trellis, max_jump_width, model_type, number_of_events, EPS, snippet, max_beam_width, rc
global source, target, data_likelihood, event_grad, dictionary_features
dictionary_features = {}
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

normalizing_decision_map = {}
pause_on_tie = False


def get_decision_given_context(theta, type, decision, context):
    global normalizing_decision_map, cache_normalizing_decision, feature_index, dictionary_features
    fired_features = get_wa_features_fired(dictionary_features=dictionary_features, type=type, context=context,
                                           decision=decision)

    theta_dot_features = sum([theta[feature_index[f]] * f_wt for f_wt, f in fired_features])

    if (type, context) in cache_normalizing_decision:
        theta_dot_normalizing_features = cache_normalizing_decision[type, context]
    else:
        normalizing_decisions = normalizing_decision_map[type, context]
        theta_dot_normalizing_features = 0
        for d in normalizing_decisions:
            d_features = get_wa_features_fired(dictionary_features=dictionary_features, type=type, context=context,
                                               decision=d)
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


def get_backwards(theta, obs_id, alpha_pi, fc=None):
    global max_jump_width, trellis, source, target
    obs = trellis[obs_id]
    src = source[obs_id]
    tar = target[obs_id]
    n = len(obs) - 1  # index of last word
    end_state = obs[n][0]
    beta_pi = {(n, end_state): 0.0}
    S = alpha_pi[(n, end_state)]  # from line 13 in pseudo code
    fc = accumulate_fc(type=E_TYPE, alpha=0.0, beta=S, e=0.0, S=S, d=BOUNDARY_START, c=BOUNDARY_START, fc=fc)
    for k in range(n, 0, -1):
        for v in obs[k]:
            tk, aj = v
            t_tok = tar[tk]
            s_tok = src[aj] if aj is not NULL else NULL
            e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)

            pb = beta_pi[(k, v)]
            fc = accumulate_fc(type=E_TYPE, alpha=alpha_pi[(k, v)], beta=beta_pi[k, v], e=e, S=S, d=t_tok, c=s_tok,
                               fc=fc)
            for u in obs[k - 1]:
                tk_1, aj_1 = u
                t_tok_1 = tar[tk_1]
                s_tok_1 = src[aj_1] if aj_1 is not NULL else NULL
                context = aj_1
                if model_type == HMM_MODEL:
                    q = get_decision_given_context(theta, T_TYPE, decision=aj, context=context)
                    fc = accumulate_fc(type=T_TYPE, alpha=alpha_pi[k - 1, u], beta=beta_pi[k, v], q=q, e=e, d=aj,
                                       c=context,
                                       S=S, fc=fc)
                else:
                    q = log(1.0 / len(obs[k]))
                    # q = 0.0

                p = q + e
                beta_p = pb + p  # The beta includes the emission probability
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = utils.logadd(beta_pi[new_pi_key], beta_p)
                    alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S
    if fc is None:
        return S, beta_pi
    else:
        return S, beta_pi, fc


def get_viterbi_and_forward(theta, obs_id):
    global max_jump_width, trellis, source, target
    src = source[obs_id]
    tar = target[obs_id]
    obs = trellis[obs_id]
    start_state = obs[0][0]
    pi = {(0, start_state): 0.0}
    alpha_pi = {(0, start_state): 0.0}
    arg_pi = {(0, start_state): []}
    for k in range(1, len(obs)):  # the words are numbered from 1 to n, 0 is special start character
        for v in obs[k]:  # [1]:
            max_prob_to_bt = {}
            sum_prob_to_bt = []
            for u in obs[k - 1]:  # [1]:
                tk, aj = v
                tk_1, aj_1 = u
                t_tok = tar[tk]
                s_tok = src[aj] if aj is not NULL else NULL
                t_tok_1 = tar[tk_1]
                s_tok_1 = src[aj_1] if aj_1 is not NULL else NULL
                if model_type == HMM_MODEL:
                    context = aj_1
                    q = get_decision_given_context(theta, T_TYPE, decision=aj, context=context)
                else:
                    q = log(1.0 / len(obs[k]))
                    # q = 0.0

                e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)

                p = pi[(k - 1, u)] + q + e
                alpha_p = alpha_pi[(k - 1, u)] + q + e
                if len(arg_pi[(k - 1, u)]) == 0:
                    bt = [u]
                else:
                    bt = [arg_pi[(k - 1, u)], u]
                max_prob_to_bt[p] = bt
                sum_prob_to_bt.append(alpha_p)
            smp = sorted(max_prob_to_bt)
            if pause_on_tie and len(smp) > 1:
                if smp[-1] == smp[-2]:
                    print 'breaking ties'
                    pdb.set_trace()
            max_bt = max_prob_to_bt[max(max_prob_to_bt)]
            new_pi_key = (k, v)
            pi[new_pi_key] = max(max_prob_to_bt)
            # print 'mu   ', new_pi_key, '=', pi[new_pi_key], exp(pi[new_pi_key])
            alpha_pi[new_pi_key] = utils.logadd_of_list(sum_prob_to_bt)
            # print 'alpha', new_pi_key, '=', alpha_pi[new_pi_key], exp(alpha_pi[new_pi_key])
            arg_pi[new_pi_key] = max_bt

    max_bt = max_prob_to_bt[max(max_prob_to_bt)]
    max_p = max(max_prob_to_bt)
    max_bt = utils.flatten_backpointers(max_bt)
    return max_bt, max_p, alpha_pi


def reset_fractional_counts():
    global fractional_counts, cache_normalizing_decision, number_of_events
    fractional_counts = {}  # dict((k, float('-inf')) for k in conditional_arc_index)
    cache_normalizing_decision = {}
    number_of_events = 0


def accumulate_fc(type, alpha, beta, d, S, c=None, k=None, q=None, e=None, fc=None):
    global fractional_counts
    if type == T_TYPE:
        update = alpha + q + e + beta - S
        if fc is None:
            fractional_counts[T_TYPE, d, c] = utils.logadd(update,
                                                           fractional_counts.get((T_TYPE, d, c,), float('-inf')))
        else:
            fc[T_TYPE, d, c] = utils.logadd(update, fc.get((T_TYPE, d, c,), float('-inf')))
    elif type == E_TYPE:
        update = alpha + beta - S  # the emission should be included in alpha
        if fc is None:
            fractional_counts[E_TYPE, d, c] = utils.logadd(update,
                                                           fractional_counts.get((E_TYPE, d, c,), float('-inf')))
        else:
            fc[E_TYPE, d, c] = utils.logadd(update, fc.get((E_TYPE, d, c,), float('-inf')))
    else:
        raise "Wrong type"
    if fc is not None:
        return fc


def get_likelihood(theta, batch=None, display=True):
    assert isinstance(theta, np.ndarray)
    assert len(theta) == len(feature_index)
    global trellis, data_likelihood, rc
    reset_fractional_counts()
    data_likelihood = 0.0

    if batch is None:
        batch = range(0, len(trellis))

    for idx in batch:
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(theta, idx)
        S, beta_pi = get_backwards(theta, idx, alpha_pi)
        # print 'p(t|s) for ', idx, ':', S, max_bt
        data_likelihood += S
    reg = np.sum(theta ** 2)
    ll = data_likelihood - (rc * reg)
    if display:
        print 'log likelihood:', ll
    return -ll


def get_likelihood_with_expected_counts_fwd(theta, batch=None, display=False):
    global cache_normalizing_decision
    cache_normalizing_decision = {}
    sum_likelihood = 0.0
    if batch is None:
        batch = range(0, len(trellis))

    for idx in batch:
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(theta, idx)
        obs = trellis[idx]
        n = len(obs) - 1  # index of last word
        end_state = obs[n][0]
        S = alpha_pi[(n, end_state)]  # from line 13 in pseudo code
        sum_likelihood += S

    reg = np.sum(theta ** 2)
    sum_likelihood -= (rc * reg)
    if display:
        print '\tec log likelihood:', sum_likelihood
    return -sum_likelihood


def get_likelihood_with_expected_counts(theta, batch=None, display=False):
    global fractional_counts
    sum_likelihood = 0.0
    for event in fractional_counts:
        (t, d, c) = event
        A_dct = exp(fractional_counts[event])
        a_dct = get_decision_given_context(theta=theta, type=t, decision=d, context=c)
        sum_likelihood += A_dct * a_dct
    reg = np.sum(theta ** 2)
    sum_likelihood -= (rc * reg)
    if display:
        print '\tec log likelihood:', sum_likelihood
    return -sum_likelihood


def get_gradient(theta, batch=None, display=True):
    global fractional_counts, feature_index, event_grad, rc, dictionary_features
    assert len(theta) == len(feature_index)
    event_grad = {}
    for event_j in fractional_counts:
        (t, dj, cj) = event_j
        f_val, f = get_wa_features_fired(dictionary_features=dictionary_features, type=t, context=cj, decision=dj)[0]
        a_dp_ct = exp(get_decision_given_context(theta, decision=dj, context=cj, type=t)) * f_val
        sum_feature_j = 0.0
        norm_events = [(t, dp, cj) for dp in normalizing_decision_map[t, cj]]
        for event_i in norm_events:
            A_dct = exp(fractional_counts.get(event_i, 0.0))
            if event_i == event_j:
                (ti, di, ci) = event_i
                fj, f = \
                    get_wa_features_fired(dictionary_features=dictionary_features, type=ti, context=ci, decision=di)[0]
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
    global EPS
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


def write_logs(theta, current_iter):
    global max_beam_width, max_jump_width, trellis, feature_index, fractional_counts
    feature_val_typ = 'bin' if options.feature_values is None else 'real'
    name_prefix = '.'.join(
        ['sp', options.algorithm, str(rc), model_type, feature_val_typ])
    if current_iter is not None:
        name_prefix += '.' + str(current_iter)
    write_weights(theta, name_prefix + '.' + options.output_weights, feature_index)
    write_probs(theta, name_prefix + '.' + options.output_probs, fractional_counts, get_decision_given_context)

    if options.source_test is not None and options.target_test is not None:
        source_test = [s.strip().split() for s in open(options.source_test, 'r').readlines()]
        target_test = [t.strip().split() for t in open(options.target_test, 'r').readlines()]
        trellis = populate_trellis(source_test, target_test, max_jump_width, max_beam_width)

    write_alignments(theta, name_prefix + '.' + options.output_alignments, trellis, get_viterbi_and_forward)
    write_alignments_col(theta, name_prefix + '.' + options.output_alignments, trellis, get_viterbi_and_forward)
    write_alignments_col_tok(theta, name_prefix + '.' + options.output_alignments, trellis, source_test, target_test,
                             get_viterbi_and_forward)


if __name__ == "__main__":
    trellis = []

    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/toy.fr")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/toy.en")
    opt.add_option("--tt", dest="target_test", default="experiment/data/toy.fr")
    opt.add_option("--ts", dest="source_test", default="experiment/data/toy.en")
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
    opt.add_option("-m", dest="model", default=IBM_MODEL_1, help="'model1' or 'hmm'")
    (options, _) = opt.parse_args()
    rc = float(options.regularization_coeff)
    model_type = options.model
    source, source_types = load_corpus_file(options.source_corpus)
    target, target_types = load_corpus_file(options.target_corpus)
    trellis = populate_trellis(source, target, max_jump_width, max_beam_width)
    dictionary_features = load_dictionary_features(options.dict_features)

    events_to_features, features_to_events, feature_index, feature_counts, event_index, event_to_event_index, event_counts, normalizing_decision_map, du = populate_features(
        trellis, source, target, model_type, dictionary_features)

    snippet = "#" + str(opt.values) + "\n"
    init_theta = initialize_theta(options.input_weights, feature_index)
    ll = get_likelihood(init_theta)

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
            old_e = float('-inf')
            converged = False
            iterations = 0
            while not converged and iterations < 5:
                t1 = minimize(get_likelihood_with_expected_counts, theta, method='L-BFGS-B', jac=get_gradient, tol=1e-2,
                              options={'maxiter': 20})
                theta = t1.x
                new_e = get_likelihood(theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 2) == 0.0
                old_e = new_e
                iterations += 1
    elif options.algorithm == "SGD":
        batch_size = len(trellis)
        theta = initialize_theta(options.input_weights, feature_index)
        get_likelihood(theta, display=True)
        reset_fractional_counts()
        batch_idxs = np.array_split(range(len(trellis)), len(trellis) / batch_size)
        for iter in xrange(3):
            random.shuffle(batch_idxs)
            b_id = 0
            for batch_idx in batch_idxs:
                print b_id
                b_id += 1
                t1 = minimize(get_likelihood, theta, method='L-BFGS-B', jac=get_gradient, args=(batch_idx, False),
                              tol=1e-5,
                              options={'maxfun': 15})
                theta = t1.x
            get_likelihood(theta, display=True)
    else:
        print 'wrong algorithm option'
        exit()

    if options.test_gradient.lower() == "true":
        pass
    else:
        write_logs(theta, current_iter=None)
