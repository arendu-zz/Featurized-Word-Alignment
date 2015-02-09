__author__ = 'arenduchintala'

from optparse import OptionParser
from math import exp, log
import sys
import multiprocessing
from multiprocessing import Pool
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
global trellis, max_jump_width, model_type, number_of_events, snippet, max_beam_width, rc
global source, target, data_likelihood, event_grad, feature_index, event_index, itercount
global events_per_trellis, event_to_event_index, has_pos, event_counts, du

snippet = ''
rc = 0.25
has_pos = False
itercount = 0
event_grad = {}
data_likelihood = 0.0
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
        log_prob = 0.0  # this happens if we truncate the LBFGS alg with maxiter
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

                p = q + e
                beta_p = pb + p  # The beta includes the emission probability
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = utils.logadd(beta_pi[new_pi_key], beta_p)
                    alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S
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

                e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)

                p = pi[(k - 1, u)] + q + e
                alpha_p = alpha_pi[(k - 1, u)] + q + e
                if len(arg_pi[(k - 1, u)]) == 0:
                    bt = [u]
                else:
                    bt = [arg_pi[(k - 1, u)], u]
                max_prob_to_bt[p] = bt
                sum_prob_to_bt.append(alpha_p)

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
    fractional_counts = {}
    cache_normalizing_decision = {}
    number_of_events = 0


def accumulate_fc(type, alpha, beta, d, S, c=None, k=None, q=None, e=None, fc=None):
    if type == T_TYPE:
        update = alpha + q + e + beta - S
        fc[T_TYPE, d, c] = utils.logadd(update, fc.get((T_TYPE, d, c,), float('-inf')))
    elif type == E_TYPE:
        update = alpha + beta - S  # the emission should be included in alpha
        fc[E_TYPE, d, c] = utils.logadd(update, fc.get((E_TYPE, d, c,), float('-inf')))
    else:
        raise "Wrong type"
    return fc


def batch_likelihood(theta, batch):
    dl = 0.0
    batch_fc = {}
    for idx in batch:
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(theta, idx)
        S, beta_pi, batch_fc = get_backwards(theta, idx, alpha_pi, batch_fc)
        dl += S
    return dl, batch_fc


def batch_accumilate_likelihood(result):
    global data_likelihood, fractional_counts
    data_likelihood += result[0]
    fc = result[1]
    for k in fc:
        fractional_counts[k] = utils.logadd(fc[k], fractional_counts.get(k, float('-inf')))


def get_likelihood(theta, display=True):
    assert isinstance(theta, np.ndarray)
    assert len(theta) == len(feature_index)
    global trellis, data_likelihood, rc, itercount, N
    reset_fractional_counts()
    data_likelihood = 0.0
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count)  # uses all available CPUs
    full = range(0, len(trellis))
    batches = np.array_split(full, cpu_count)
    for batch in batches:
        pool.apply_async(batch_likelihood, args=(theta, batch), callback=batch_accumilate_likelihood)
    pool.close()
    pool.join()
    reg = np.sum(theta ** 2)
    ll = (data_likelihood - (rc * reg))
<<<<<<< HEAD
    e1 = get_decision_given_context(theta, E_TYPE, decision='.', context=NULL)
    e2 = get_decision_given_context(theta, E_TYPE, decision='.', context='.')
    print itercount, 'log likelihood:', ll, 'p(.|NULL)', e1, 'p(.|.)', e2 
    itercount += 1 
=======
    if display:
        print itercount, 'log likelihood:', ll
    itercount += 1
>>>>>>> f411800267e716b637dbc33d0915f0266e4d368e
    return -ll


def batch_gradient(theta, batch_fractional_counts):
    global event_index
    eg = {}
    for idx in batch_fractional_counts:
        (t, dj, cj) = event_index[idx]
        f_val, f = FE.get_wa_features_fired(type=t, context=cj, decision=dj)[0]  # TODO: this only works in basic feat
        a_dp_ct = exp(get_decision_given_context(theta, decision=dj, context=cj, type=t)) * f_val
        sum_feature_j = 0.0
        norm_events = [(t, dp, cj) for dp in normalizing_decision_map[t, cj]]

        for event_i in norm_events:
            A_dct = exp(fractional_counts.get(event_i, 0.0))
            if event_i == event_index[idx]:
                (ti, di, ci) = event_i
                fj, f = FE.get_wa_features_fired(type=ti, context=ci, decision=di)[0]  # TODO: this only works in basic
            else:
                fj = 0.0
            sum_feature_j += A_dct * (fj - a_dp_ct)
        eg[(t, dj, cj)] = sum_feature_j

    return eg


def batch_accumilate_gradient(result):
    global event_grad
    for event_j in result:
        if event_j in event_grad:
            raise 'should this happen?'
        else:
            event_grad[event_j] = result[event_j]


def get_gradient(theta):
    global fractional_counts, event_index, feature_index, event_grad, rc, N
    assert len(theta) == len(feature_index)
    event_grad = {}
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count)  # uses all available CPUs
    batches_fractional_counts = np.array_split(range(len(event_index)), cpu_count)
    for batch_of_fc in batches_fractional_counts:
        pool.apply_async(batch_gradient, args=(theta, batch_of_fc), callback=batch_accumilate_gradient)
    pool.close()
    pool.join()
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


def batch_likelihood_with_expected_counts(theta, batch):
    global event_index
    batch_sum_likelihood = 0.0
    for idx in batch:
        event = event_index[idx]
        (t, d, c) = event_index[idx]
        A_dct = exp(fractional_counts[event])
        a_dct = get_decision_given_context(theta=theta, type=t, decision=d, context=c)
        batch_sum_likelihood += A_dct * a_dct
    return batch_sum_likelihood


def batch_accumilate_likelihood_with_expected_counts(results):
    global data_likelihood
    data_likelihood += results


def get_likelihood_with_expected_counts(theta, display=True):
    global fractional_counts, data_likelihood, event_index
    data_likelihood = 0.0
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count)  # uses all available CPUs
    batches_fractional_counts = np.array_split(range(len(event_index)), cpu_count)
    for batch_of_fc in batches_fractional_counts:
        pool.apply_async(batch_likelihood_with_expected_counts, args=(theta, batch_of_fc),
                         callback=batch_accumilate_likelihood_with_expected_counts)
    pool.close()
    pool.join()

    reg = np.sum(theta ** 2)
    data_likelihood -= (rc * reg)
<<<<<<< HEAD
=======
    # if display:
    # print '\tec:', data_likelihood
>>>>>>> f411800267e716b637dbc33d0915f0266e4d368e
    return -data_likelihood


def populate_events_per_trellis():
    global event_index, trellis, events_per_trellis
    events_per_trellis = []
    for obs_id, obs in enumerate(trellis):
        events_observed = []
        obs = trellis[obs_id]
        src = source[obs_id]
        tar = target[obs_id]
        for k in range(1, len(obs)):  # the words are numbered from 1 to n, 0 is special start character
            for v in obs[k]:  # [1]:
                for u in obs[k - 1]:  # [1]:
                    tk, aj = v
                    tk_1, aj_1 = u
                    t_tok = tar[tk]
                    s_tok = src[aj] if aj is not NULL else NULL
                    if model_type == HMM_MODEL:
                        context = aj_1
                        ei = event_to_event_index[(T_TYPE, aj, context)]
                        event_observed.append(ei)
                    else:
                        pass

                    ei = event_to_event_index[(E_TYPE, t_tok, s_tok)]
                    events_observed.append(ei)
        events_per_trellis.append(list(set(events_observed)))


<<<<<<< HEAD
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


=======
>>>>>>> f411800267e716b637dbc33d0915f0266e4d368e
def gradient_check_em():
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
    global feature_index
    init_theta = initialize_theta(None, True)
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
        ['mp', options.algorithm, str(rc), model_type, feature_val_typ])
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


def batch_sgd(obs_id, sgd_theta, sum_square_grad):
    # print _, obs_id
    eo = events_per_trellis[obs_id]
    eg = batch_gradient(sgd_theta, eo)
    gdu = np.array([float('inf')] * len(sgd_theta))
    grad = np.zeros(np.shape(sgd_theta))  # -2 * rc * theta  # l2 regularization with lambda 0.5
    for e in eg:
        feats = events_to_features[e]
        for f in feats:
            grad[feature_index[f]] += eg[e]
            gdu[feature_index[f]] = du[feature_index[f]]

    grad_du = -2 * rc * np.divide(sgd_theta, gdu)

    grad += grad_du
    sum_square_grad += (grad ** 2)
    eta_t = eta0 / np.sqrt(I + sum_square_grad)
    sgd_theta += np.multiply(eta_t, grad)
    return obs_id


def batch_sgd_accumilate(obs_id):
    # print obs_id
    pass


import sharedmem

if __name__ == "__main__":
    trellis = []
    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/toy.fr")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/toy.en")
    opt.add_option("--tt", dest="target_test", default="experiment/data/toy.fr")
    opt.add_option("--ts", dest="source_test", default="experiment/data/toy.en")

    opt.add_option("--il", dest="intermediate_log", default="0")
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

    source = [s.strip().split() for s in open(options.source_corpus, 'r').readlines()]
    target = [s.strip().split() for s in open(options.target_corpus, 'r').readlines()]

    trellis = populate_trellis(source, target, max_jump_width, max_beam_width)
    events_to_features, features_to_events, feature_index, feature_counts, event_index, event_to_event_index, event_counts, normalizing_decision_map = populate_features(
        trellis, source, target, model_type)

    FE.load_feature_values(options.feature_values)
    snippet = "#" + str(opt.values) + "\n"

    if options.algorithm == "LBFGS":
        if options.test_gradient.lower() == "true":
            gradient_check_lbfgs()
        else:
            print 'skipping gradient check...'
<<<<<<< HEAD
            init_theta = initialize_theta(options.input_weights)
=======
            init_theta = initialize_theta(options.input_weights, feature_index)
>>>>>>> f411800267e716b637dbc33d0915f0266e4d368e
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
                t1 = minimize(get_likelihood_with_expected_counts, theta, method='L-BFGS-B', jac=get_gradient, tol=1e-3,
                              options={'maxiter': 20})
                theta = t1.x
                new_e = get_likelihood(theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 1) == 0.0
                old_e = new_e
                iterations += 1
    elif options.algorithm == "EM-SGD":
        if options.test_gradient.lower() == "true":
            gradient_check_em()
        else:
            print 'skipping gradient check...'
            print 'populating events per trellis...'
            populate_events_per_trellis()
            print 'done...'
            theta = initialize_theta(options.input_weights, feature_index)
            new_e = get_likelihood(theta)
            exp_new_e = get_likelihood_with_expected_counts(theta)
            old_e = float('-inf')
            converged = False
            iterations = 0
            ids = range(len(trellis))

            while not converged and iterations < 10:
                eta0 = 1.0
                sum_square_grad = np.zeros(np.shape(theta))
                I = 1.0
                for _ in range(2):
                    random.shuffle(ids)
                    for obs_id in ids:
                        print _, obs_id
                        event_observed = events_per_trellis[obs_id]
                        eg = batch_gradient(theta, event_observed)
                        grad = -2 * rc * theta  # l2 regularization with lambda 0.5
                        for e in eg:
                            feats = events_to_features[e]
                            for f in feats:
                                grad[feature_index[f]] += eg[e]
                        sum_square_grad += (grad ** 2)
                        eta_t = eta0 / np.sqrt(I + sum_square_grad)
                        theta += np.multiply(eta_t, grad)

                new_e = get_likelihood(theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 2) == 0.0
                old_e = new_e
                iterations += 1

    elif options.algorithm == "EM-SGD-PARALLEL":
        if options.test_gradient.lower() == "true":
            gradient_check_em()
        else:
            print 'skipping gradient check...'
            print 'populating events per trellis...'
            populate_events_per_trellis()
            print 'done...'

            init_theta = initialize_theta(options.input_weights, feature_index)
            shared_sgd_theta = sharedmem.zeros(np.shape(init_theta))
            shared_sgd_theta += init_theta
            new_e = get_likelihood(shared_sgd_theta)
            exp_new_e = get_likelihood_with_expected_counts(shared_sgd_theta)
            old_e = float('-inf')
            converged = False
            iterations = 0
            ids = range(len(trellis))
            while not converged and iterations < 5:
                eta0 = 1.0
                shared_sum_squared_grad = sharedmem.zeros(np.shape(shared_sgd_theta))
                I = 1.0
                for _ in range(2):
                    random.shuffle(ids)

                    cpu_count = multiprocessing.cpu_count()
                    pool = Pool(processes=cpu_count)
                    for obs_id in ids:
                        pool.apply_async(batch_sgd, args=(obs_id, shared_sgd_theta, shared_sum_squared_grad),
                                         callback=batch_sgd_accumilate)
                    pool.close()
                    pool.join()
                    """
                    for obs_id in ids:
                        batch_sgd(obs_id)
                    """
                new_e = get_likelihood(shared_sgd_theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 2) == 0.0
                old_e = new_e
                iterations += 1
            theta = shared_sgd_theta
    else:
        print 'wrong option for algorithm...'
        exit()

    if options.test_gradient.lower() == "true":
        pass
    else:
        write_logs(theta, current_iter=None)
