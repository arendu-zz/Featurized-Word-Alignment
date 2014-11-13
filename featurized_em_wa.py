__author__ = 'arenduchintala'

from optparse import OptionParser
from math import exp, log, sqrt
from DifferentiableFunction import DifferentiableFunction
import numpy as np
import FeatureEng as FE
import utils
from pprint import pprint
from collections import defaultdict
import random
import copy

global BOUNDARY_START, END_STATE, SPLIT, E_TYPE, T_TYPE, IBM_MODEL_1, HMM_MODEL, possible_states
global cache_normalizing_decision, features_to_events, events_to_features, normalizing_decision_map
global trellis, max_jump_width, model_type, number_of_events, EPS, snippet
snippet = ''
EPS = 1e-8
IBM_MODEL_1 = "model1"
HMM_MODEL = "hmm"
max_jump_width = 10  # creates a span of +/- span centered around current token
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
possible_states = {}
possible_obs = {}
normalizing_decision_map = {}


def get_jump(aj, aj_1):
    if aj != NULL and aj_1 != NULL:
        jump = abs(aj_1 - aj)
        assert jump <= 2 * max_jump_width + 1
    else:
        jump = NULL
    return jump


def populate_features():
    global trellis, feature_index
    for treli in trellis:
        for idx in treli:
            for t_idx, t_tok, s_idx, s_tok, L in treli[idx]:
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
                for f in ff_e:
                    feature_index[f] = feature_index.get(f, 0) + 1
                    ca2f = events_to_features.get(emission_event, set([]))
                    ca2f.add(f)
                    events_to_features[emission_event] = ca2f
                    f2ca = features_to_events.get(f, set([]))
                    f2ca.add(emission_event)
                    features_to_events[f] = f2ca

                if idx > 0 and model_type == HMM_MODEL:
                    for prev_t_idx, prev_t_tok, prev_s_idx, prev_s_tok, L in treli[idx - 1]:
                        # an arc in an hmm can have the following information:
                        # prev_s_idx, prev_s_tok, s_idx, s_tok, t_idx, t_tok
                        prev_s = (prev_s_idx, prev_s_tok)
                        curr_s = (s_idx, s_tok)
                        curr_t = (t_idx, t_tok)
                        # arc = (prev_s, curr_s, curr_t)

                        # transition features

                        # jump = get_jump(s_idx, prev_s_idx)
                        transition_context = (prev_s_idx, L)
                        transition_decision = s_idx
                        transition_event = (T_TYPE, transition_decision, transition_context)
                        ff_t = FE.get_wa_features_fired(type=T_TYPE, decision=transition_decision,
                                                        context=transition_context)

                        ndm = normalizing_decision_map.get((T_TYPE, transition_context), set([]))
                        ndm.add(transition_decision)
                        normalizing_decision_map[T_TYPE, transition_context] = ndm
                        for f in ff_t:
                            feature_index[f] = feature_index.get(f, 0) + 1
                            ca2f = events_to_features.get(transition_event, set([]))
                            ca2f.add(f)
                            events_to_features[transition_event] = ca2f
                            f2ca = features_to_events.get(f, set([]))
                            f2ca.add(transition_event)
                            features_to_events[f] = f2ca


def get_transition_no_feature(jump):
    p = utils.normpdf(jump, 1.0, 3.0)  # TODO: what should the variance be?
    return log(p)


def get_decision_given_context(theta, type, decision, context):
    global normalizing_decision_map, cache_normalizing_decision
    fired_features = FE.get_wa_features_fired(type=type, context=context, decision=decision)

    theta_dot_features = sum([theta[f] for f in fired_features])
    # TODO: weights theta are initialized to 0.0
    # TODO: this initialization should be done in a better way
    if (type, context) in cache_normalizing_decision:
        theta_dot_normalizing_features = cache_normalizing_decision[type, context]
    else:
        normalizing_decisions = normalizing_decision_map[type, context]
        theta_dot_normalizing_features = float('-inf')
        for d in normalizing_decisions:
            d_features = FE.get_wa_features_fired(type=type, context=context, decision=d)
            theta_dot_normalizing_features = utils.logadd(theta_dot_normalizing_features,
                                                          sum([theta[f] for f in d_features]))

        cache_normalizing_decision[type, context] = theta_dot_normalizing_features
    log_prob = round(theta_dot_features - theta_dot_normalizing_features, 10)
    if log_prob > 0.0:
        # print log_prob, type, decision, context
        # pdb.set_trace()
        log_prob = 0.0
        # raise Exception
    return log_prob


def get_possible_states(o):
    if o == BOUNDARY_START:
        return [BOUNDARY_START]
    elif o == BOUNDARY_END:
        return [BOUNDARY_END]
    # elif o in possible_states:
    # return list(possible_states[o])
    else:
        return list(possible_states[ALL] - set([BOUNDARY_START, BOUNDARY_END]))


def get_backwards(theta, obs, alpha_pi):
    n = len(obs) - 1  # index of last word
    end_state = obs[n][0]
    beta_pi = {(n, end_state): 0.0}
    S = alpha_pi[(n, end_state)]  # from line 13 in pseudo code
    accumulate_fc(type=E_TYPE, alpha=0.0, beta=S, e=0.0, S=S, d=BOUNDARY_START, c=BOUNDARY_START)
    for k in range(n, 0, -1):
        for v in obs[k]:
            tk, t_tok, aj, s_tok, L = v
            e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
            pb = beta_pi[(k, v)]
            accumulate_fc(type=E_TYPE, alpha=alpha_pi[(k, v)], beta=beta_pi[k, v], e=e, S=S, d=t_tok, c=s_tok)
            for u in obs[k - 1]:
                tk_1, t_tok_1, aj_1, s_tok_1, L = u
                context = (aj_1, L)
                if model_type == HMM_MODEL:
                    q = get_decision_given_context(theta, T_TYPE, decision=aj, context=context)
                    accumulate_fc(type=T_TYPE, alpha=alpha_pi[k - 1, u], beta=beta_pi[k, v], q=q, e=e, d=aj,
                                  c=context,
                                  S=S)
                else:
                    q = log(1.0 / len(obs[k]))

                p = q + e
                beta_p = pb + p  # The beta includes the emission probability
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = utils.logadd(beta_pi[new_pi_key], beta_p)
                    # print 'beta     ', new_pi_key, '=', beta_pi[new_pi_key], exp(beta_pi[new_pi_key])
                    alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S

                    # print 'beta     ', new_pi_key, '=', beta_pi[new_pi_key], exp(beta_pi[new_pi_key])

    return S, beta_pi


def get_viterbi_and_forward(theta, obs):
    global max_jump_width
    start_state = obs[0][0]
    pi = {(0, start_state): 0.0}
    alpha_pi = {(0, start_state): 0.0}
    arg_pi = {(0, start_state): []}
    for k in range(1, len(obs)):  # the words are numbered from 1 to n, 0 is special start character
        for v in obs[k]:  # [1]:
            max_prob_to_bt = {}
            sum_prob_to_bt = []
            for u in obs[k - 1]:  # [1]:
                tk, t_tok, aj, s_tok, L = v
                tk_1, t_tok_1, aj_1, s_tok_1, L = u
                if model_type == HMM_MODEL:
                    context = (aj_1, L)
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
    fractional_counts = {}  # dict((k, float('-inf')) for k in conditional_arc_index)
    cache_normalizing_decision = {}
    number_of_events = 0


def accumulate_fc(type, alpha, beta, d, S, c=None, k=None, q=None, e=None):
    global fractional_counts, number_of_events
    number_of_events += 1
    if type == T_TYPE:
        update = alpha + q + e + beta - S
        fractional_counts[T_TYPE, d, c] = utils.logadd(update, fractional_counts.get((T_TYPE, d, c,), float('-inf')))
    elif type == E_TYPE:
        update = alpha + beta - S  # the emission should be included in alpha
        fractional_counts[E_TYPE, d, c] = utils.logadd(update, fractional_counts.get((E_TYPE, d, c,), float('-inf')))
    else:
        raise "Wrong type"


def write_probs(theta, save_probs):
    write_probs = open(save_probs, 'w')
    write_probs.write(snippet)
    for fc in sorted(fractional_counts):
        (t, d, c) = fc
        prob = get_decision_given_context(theta, type=t, decision=d, context=c)
        str_t = reduce(lambda a, d: str(a) + '\t' + str(d), fc, '')
        write_probs.write(str_t.strip() + '\t' + str(round(prob, 5)) + '' + "\n")
    write_probs.flush()
    write_probs.close()
    print 'wrote weights to:', save_probs


def write_weights(theta, save_weights):
    global trellis
    write_theta = open(save_weights, 'w')
    write_theta.write(snippet)
    for t in sorted(theta):
        str_t = reduce(lambda a, d: str(a) + '\t' + str(d), t, '')
        write_theta.write(str_t.strip() + '\t' + str(theta[t]) + '' + "\n")
    write_theta.flush()
    write_theta.close()
    print 'wrote weights to:', save_weights


def write_alignments(theta, save_align):
    global trellis
    write_align = open(save_align, 'w')
    write_align.write(snippet)
    for idx, obs in enumerate(trellis[:]):
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(theta, obs)
        w = ' '.join([str(tar_i - 1) + '-' + str(src_i - 1) for src_i, src, tar_i, tar, L in max_bt if
                      (tar_i != NULL and tar_i > 0 and src_i > 0)])
        write_align.write(w + '\n')
    write_align.flush()
    write_align.close()
    print 'wrote alignments to:', save_align


def get_likelihood(theta, start_idx=None, end_idx=None):
    global trellis
    reset_fractional_counts()
    data_likelihood = 0.0
    st = 0 if start_idx == None else start_idx
    end = len(trellis) if end_idx == None else end_idx
    for idx, obs in enumerate(trellis[st:end]):
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(theta, obs)
        S, beta_pi = get_backwards(theta, obs, alpha_pi)
        data_likelihood += S

    reg = sum([t ** 2 for t in theta.values()])
    ll = data_likelihood - (0.5 * reg)
    print 'log likelihood:', ll
    return ll


def get_likelihood_with_expected_counts(theta):
    global fractional_counts
    sum_likelihood = 0.0
    for event in fractional_counts:
        (t, d, c) = event
        A_dct = exp(fractional_counts[event])
        a_dct = get_decision_given_context(theta=theta, type=t, decision=d, context=c)
        # print event, A_dct, a_dct, A_dct * a_dct

        sum_likelihood += A_dct * a_dct
        # print 'sl', sum_likelihood
    reg = sum([t ** 2 for t in theta.values()])
    return sum_likelihood - (0.5 * reg)


def get_gradient(theta):
    global fractional_counts
    event_grad = {}
    for event_j in fractional_counts.keys():
        (t, dj, cj) = event_j
        a_dp_ct = exp(get_decision_given_context(theta, decision=dj, context=cj, type=t))
        sum_feature_j = 0.0
        norm_events = [(t, dp, cj) for dp in normalizing_decision_map[t, cj]]
        for event_i in norm_events:
            A_dct = exp(fractional_counts.get(event_i, 0.0))
            fj = 1.0 if event_i == event_j else 0.0
            sum_feature_j += A_dct * (fj - a_dp_ct)
        event_grad[event_j] = sum_feature_j  # - abs(theta[event_j])  # this is the regularizing term

    grad = {}
    for e in event_grad:
        feats = events_to_features[e]
        for f in feats:
            grad[f] = grad.get(f, 0.0) + event_grad[e]

    for f in grad:
        grad[f] -= theta[f]  # for l2 regularization with lambda 0.5
    return grad


def populate_trellis(source_corpus, target_corpus):
    global trellis, max_jump_width
    for s_sent, t_sent in zip(source_corpus, target_corpus):
        t_sent.insert(0, BOUNDARY_START)
        s_sent.insert(0, BOUNDARY_START)
        t_sent.append(BOUNDARY_END)
        s_sent.append(BOUNDARY_END)
        current_trellis = {}
        for t_idx, t_tok in enumerate(t_sent):
            if t_tok == BOUNDARY_START:
                current_trellis[t_idx] = [(t_idx, BOUNDARY_START, t_idx, BOUNDARY_START, len(s_sent))]
            elif t_tok == BOUNDARY_END:
                current_trellis[t_idx] = [(t_idx, BOUNDARY_END, len(s_sent) - 1, BOUNDARY_END, len(s_sent))]
            else:
                start = t_idx - (max_jump_width) if t_idx - (max_jump_width) >= 0 else 0
                end = t_idx + (max_jump_width) + 1
                assert end - start <= 2 * max_jump_width + 1
                current_trellis[t_idx] = [(t_idx, t_tok, s_idx + start, s_tok, len(s_sent)) for s_idx, s_tok in
                                          enumerate(s_sent[start:end]) if
                                          (s_tok != BOUNDARY_START and s_tok != BOUNDARY_END)]
                # current_trellis[t_idx] += [(t_idx, t_tok, NULL, NULL, len(s_sent))]
        trellis.append(current_trellis)


def gradient_check_em():
    global EPS
    init_theta = dict((k, np.random.uniform(-1.0, 1.0)) for k in feature_index)
    f_approx = {}
    for i in init_theta:
        theta_plus = copy.deepcopy(init_theta)
        theta_minus = copy.deepcopy(init_theta)
        theta_plus[i] = init_theta[i] + EPS
        get_likelihood(theta_plus)  # updates fractional counts
        val_plus = get_likelihood_with_expected_counts(theta_plus)
        theta_minus[i] = init_theta[i] - EPS
        get_likelihood(theta_minus)  # updates fractional counts
        val_minus = get_likelihood_with_expected_counts(theta_minus)
        f_approx[i] = (val_plus - val_minus) / (2 * EPS)

    my_grad = get_gradient(init_theta)
    diff = []
    for k in sorted(my_grad):
        diff.append(abs(my_grad[k] - f_approx[k]))
        print str(round(my_grad[k] - f_approx[k], 3)).center(10), str(
            round(my_grad[k], 5)).center(10), \
            str(round(f_approx[k], 5)).center(10), k
    print 'component difference:', round(sum(diff), 3), \
        'cosine similarity:', utils.cosine_sim(f_approx, my_grad), \
        ' sign difference', utils.sign_difference(f_approx, my_grad)


def gradient_check_lbfgs():
    global EPS
    init_theta = dict((k, np.random.uniform(-1.0, 1.0)) for k in feature_index)
    chk_grad = utils.gradient_checking(init_theta, EPS, get_likelihood)
    my_grad = get_gradient(init_theta)
    diff = []
    for k in sorted(my_grad):
        diff.append(abs(my_grad[k] - chk_grad[k]))
        print str(round(my_grad[k] - chk_grad[k], 5)).center(10), str(
            round(my_grad[k], 5)).center(10), \
            str(round(chk_grad[k], 5)).center(10), k

    print 'component difference:', round(sum(diff), 3), \
        'cosine similarity:', utils.cosine_sim(chk_grad, my_grad), \
        ' sign difference', utils.sign_difference(chk_grad, my_grad)


if __name__ == "__main__":
    trellis = []
    possible_states = defaultdict(set)
    possible_obs = defaultdict(set)
    opt = OptionParser()

    opt.add_option("-t", dest="target_corpus", default="data/small/en-toy")
    opt.add_option("-s", dest="source_corpus", default="data/small/fr-toy")
    opt.add_option("--ow", dest="output_weights", default="theta", help="extention of trained weights file")
    opt.add_option("--oa", dest="output_alignments", default="alignments", help="extension of alignments files")
    opt.add_option("--op", dest="output_probs", default="probs", help="extension of probabilities")
    opt.add_option("-g", dest="test_gradient", default="false")
    opt.add_option("-a", dest="algorithm", default="LBFGS",
                   help="use 'EM' 'LBFGS' 'SGD'")
    opt.add_option("-m", dest="model", default=IBM_MODEL_1, help="'model1' or 'hmm'")
    (options, _) = opt.parse_args()

    model_type = options.model
    source = [s.strip().split() for s in open(options.source_corpus, 'r').readlines() if s.strip() != '']
    target = [s.strip().split() for s in open(options.target_corpus, 'r').readlines() if s.strip() != '']
    populate_trellis(source, target)
    populate_features()

    snippet = "#" + str(opt.values) + "\n"
    print snippet

    if options.algorithm == "LBFGS":

        if options.test_gradient.lower() == "true":
            gradient_check_lbfgs()
        else:
            print 'skipping gradient check...'

            init_theta = dict((k, np.random.uniform(-1.0, 1.0)) for k in feature_index)

            F = DifferentiableFunction(get_likelihood, get_gradient)
            (fopt, theta, return_status) = F.maximize(init_theta)
            get_likelihood(theta)
            write_alignments(theta, options.algorithm + '.' + model_type + '.' + options.output_alignments)
            write_weights(theta, options.algorithm + '.' + model_type + '.' + options.output_weights)
            write_probs(theta, options.algorithm + '.' + model_type + '.' + options.output_probs)

    if options.algorithm == "EM":

        if options.test_gradient.lower() == "true":
            gradient_check_em()
        else:
            print 'skipping gradient check...'

            init_theta = dict((k, np.random.uniform(-1.0, 1.0)) for k in feature_index)
            new_e = get_likelihood(init_theta)
            old_e = float('-inf')
            converged = False
            while not converged:
                F = DifferentiableFunction(get_likelihood_with_expected_counts, get_gradient)
                (fopt, theta, return_status) = F.maximize(init_theta, maxfun=5)
                new_e = get_likelihood(theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 2) == 0.0
                old_e = new_e
            write_alignments(theta, options.algorithm + '.' + model_type + '.' + options.output_alignments)
            write_weights(theta, options.algorithm + '.' + model_type + '.' + options.output_weights)
            write_probs(theta, options.algorithm + '.' + model_type + '.' + options.output_probs)

    if options.algorithm == "SGD":
        theta = dict((k, np.random.uniform(-1.0, 1.0)) for k in feature_index)
        get_likelihood(theta)
        eta = 1.0
        I = 1.0
        step_size = 0.1
        idxs = range(len(trellis))
        t = 0
        for iter in xrange(500):
            random.shuffle(idxs)
            for i in idxs:
                t += 1
                print idxs[i]
                new_ll = get_likelihood(theta, start_idx=idxs[i], end_idx=idxs[i] + 1)
                grad = get_gradient(theta)
                for g in grad:
                    theta[g] = theta[g] - (step_size * grad[g])

            print 'full ll', get_likelihood(theta)





