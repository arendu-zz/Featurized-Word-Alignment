__author__ = 'arenduchintala'

from optparse import OptionParser
from math import exp, log
import utils
from pprint import pprint
from collections import defaultdict

global BOUNDARY_STATE, END_STATE, SPLIT, E_TYPE, T_TYPE, theta, possible_states, normalizing_decision_map
BOUNDARY_STATE = "###"
SPLIT = "###/###"
E_TYPE = "emission"
E_TYPE_PRE = "prefix_feature"
E_TYPE_SUF = "suffix_feature"
T_TYPE = "transition"
S_TYPE = "state"
ALL = "ALL_STATES"
fractional_counts = {}
conditional_arcs_to_features = {}
features_to_conditional_arcs = {}
theta = {}
possible_states = defaultdict(set)
possible_obs = defaultdict(set)
normalizing_decision_map = {}


def populate_arcs_to_features(co_occurance):
    for type, d, c in co_occurance:
        fired_features = get_features_fired(type=type, decision=d, context=c)
        fs = conditional_arcs_to_features.get((type, d, c), set([]))
        fs = fs.union(fired_features)
        conditional_arcs_to_features[type, d, c] = fs
        for ff in fired_features:
            cs = features_to_conditional_arcs.get(ff, set([]))
            cs.add((type, d, c))
            features_to_conditional_arcs[ff] = cs


def accumulate_fractional_counts(fc):
    global fractional_counts
    for type, d, c in fc:
        fractional_counts[type, d, c] = utils.logadd(fc[type, d, c], fractional_counts.get((type, d, c), float('-Inf')))


def populate_normalizing_terms(co_occurance=None):
    if co_occurance is None:
        for s in possible_states[ALL]:
            normalizing_decision_map[E_TYPE, s] = possible_obs[ALL]
            normalizing_decision_map[T_TYPE, s] = possible_states[ALL]
    else:
        for type, f1, f2 in co_occurance:
            # f1 is the decision variable
            # f2 is the context variable
            n = normalizing_decision_map.get((type, f2), set([]))
            n.add(f1)
            normalizing_decision_map[type, f2] = n


def get_features_fired(type, decision, context):
    if type == E_TYPE:
        # suffix feature, prefix feature
        return [(E_TYPE_SUF, decision[-4:], context), (E_TYPE_PRE, decision[:4], context), (E_TYPE, decision, context)]
    elif type == T_TYPE:
        return [(T_TYPE, decision, context)]


def get_emission(obs, state):
    global normalizing_decision_map
    fired_features = get_features_fired(type=E_TYPE, context=state, decision=obs)
    theta_dot_features = sum([theta.get(f, 0.0) for f in fired_features])
    normalizing_decisions = normalizing_decision_map[E_TYPE, state]
    theta_dot_normalizing_features = 0.0
    for d in normalizing_decisions:
        d_features = get_features_fired(type=E_TYPE, context=state, decision=d)
        theta_dot_normalizing_features += exp(sum([theta.get(f, 0.0) for f in d_features]))
    log_prob = theta_dot_features - theta_dot_normalizing_features
    return log_prob


def get_transition(state, prev_state):
    global normalizing_decision_map
    fired_features = get_features_fired(type=T_TYPE, decision=state, context=prev_state)
    theta_dot_features = sum([theta.get(f, 0.0) for f in fired_features])
    normalizing_decisions = normalizing_decision_map[T_TYPE, prev_state]
    theta_dot_normalizing_features = 0.0
    for d in normalizing_decisions:
        d_features = get_features_fired(type=T_TYPE, decision=d, context=prev_state)
        theta_dot_normalizing_features += exp(sum([theta.get(f, 0.0) for f in d_features]))
    log_prob = theta_dot_features - theta_dot_normalizing_features
    return log_prob


def get_possible_states(o):
    if o == BOUNDARY_STATE:
        return [BOUNDARY_STATE]
    elif o in possible_states:
        return list(possible_states[ALL])
    else:
        return list(possible_states[ALL])


def get_backwards(raw_words, alpha_pi):
    words = [BOUNDARY_STATE] + raw_words + [BOUNDARY_STATE]
    n = len(words) - 1  # index of last word
    beta_pi = {(n, BOUNDARY_STATE): 0.0}
    S = alpha_pi[(n, BOUNDARY_STATE)]  # from line 13 in pseudo code
    for k in range(n, 0, -1):
        for v in get_possible_states(words[k]):
            e = get_emission(words[k], v)
            pb = beta_pi[(k, v)]

            for u in get_possible_states(words[k - 1]):
                q = get_transition(v, u)
                p = q + e
                beta_p = pb + p
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = utils.logadd(beta_pi[new_pi_key], beta_p)
                    # print 'beta     ', new_pi_key, '=', beta_pi[new_pi_key], exp(beta_pi[new_pi_key])
                '''
                if k not in posterior_bigrams:
                    posterior_bigrams[k] = [((u, v), posterior_bigram_val)]
                else:
                    posterior_bigrams[k].append(((u, v), posterior_bigram_val))
                '''

    return S, beta_pi


def get_viterbi_and_forward(raw_words):
    words = [BOUNDARY_STATE] + raw_words + [BOUNDARY_STATE]
    pi = {(0, BOUNDARY_STATE): 0.0}
    alpha_pi = {(0, BOUNDARY_STATE): 0.0}
    # pi[(0, START_STATE)] = 1.0  # 0,START_STATE
    arg_pi = {(0, BOUNDARY_STATE): []}
    for k in range(1, len(words)):  # the words are numbered from 1 to n, 0 is special start character
        for v in get_possible_states(words[k]):  # [1]:
            max_prob_to_bt = {}
            sum_prob_to_bt = []
            for u in get_possible_states(words[k - 1]):  # [1]:
                q = get_transition(v, u)
                e = get_emission(words[k], v)
                # p = pi[(k - 1, u)] * q * e
                # alpha_p = alpha_pi[(k - 1, u)] * q * e
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


def get_fractional_counts(alpha_pi, beta_pi, raw_words):
    words = [BOUNDARY_STATE] + raw_words + [BOUNDARY_STATE]
    trellis_states = alpha_pi.keys()
    trellis_states.sort()
    alpha_beta = {}
    alpha_beta_t = defaultdict(float)  # alpha times beta summed over all states at time step 't'
    time2nodes = {}
    for t, n in trellis_states:
        ns = time2nodes.get(t, [])
        ns.append(n)
        time2nodes[t] = ns
        alpha_beta[t, n] = alpha_pi[t, n] + beta_pi[t, n]
        alpha_beta_t[t] = utils.logadd(alpha_beta[t, n], alpha_beta_t.get(t, float('-Inf')))
        # print t, '\t\t', n, '\t\t', alpha_pi[t, n], '\t\t', beta_pi[t, n], '\t\t', alpha_beta[t, n]

    fc = {}
    for t, n in trellis_states:
        if t > 0:
            obs = words[t]
            # probability of being in node n at time t
            update = alpha_beta[t, n] - alpha_beta_t[t]
            acc = fc.get((S_TYPE, n, None), float('-Inf'))
            fc[S_TYPE, n, None] = utils.logadd(acc, update)

            acc = fc.get((E_TYPE, obs, n), float('-Inf'))
            fc[E_TYPE, obs, n] = utils.logadd(acc, update)
            beta_t1n = beta_pi[t, n]
            ne = get_emission(words[t], n)
            for pn in time2nodes[t - 1]:
                alpha_t0pn = alpha_pi[t - 1, pn]
                pn2n = get_transition(n, pn)
                update = (beta_t1n + alpha_t0pn + pn2n + ne) - (alpha_beta_t[t])
                acc = fc.get((T_TYPE, n, pn), float('-Inf'))
                fc[T_TYPE, n, pn] = utils.logadd(acc, update)
    return fc


if __name__ == "__main__":
    opt = OptionParser()
    opt.add_option("-i", dest="initial_train", default="data/entrain4k")
    opt.add_option("-t", dest="raw", default="data/enraw")
    (options, _) = opt.parse_args()
    possible_states[ALL] = set([])
    co_occurance = {}
    for t in open(options.initial_train, 'r').read().split(SPLIT):
        if t.strip() != '':
            obs_state = [tuple(x.split('/')) for x in t.split('\n') if x.strip() != '']
            obs_state.append((BOUNDARY_STATE, BOUNDARY_STATE))
            obs_state.insert(0, (BOUNDARY_STATE, BOUNDARY_STATE))
            for idx, (obs, state) in enumerate(obs_state[1:]):
                possible_states[obs].add(state)
                possible_states[ALL].add(state)
                prev_obs, prev_state = obs_state[idx - 1]
                possible_obs[state].add(obs)
                possible_obs[ALL].add(obs)
                co_occurance[E_TYPE, obs, state] = None
                co_occurance[T_TYPE, state, prev_state] = None
    populate_normalizing_terms(co_occurance)
    populate_arcs_to_features(co_occurance)
    possible_states[ALL].remove(BOUNDARY_STATE)

    for idx, t in enumerate(open(options.initial_train, 'r').read().split(SPLIT)[:4]):
        if t.strip() != '':
            obs_state = [tuple(x.split('/')) for x in t.split('\n') if x.strip() != '']
            obs, state = zip(*obs_state)
            obs = list(obs)
            # obs = "this is big .".split()
            max_bt, max_p, alpha_pi = get_viterbi_and_forward(obs)
            # print(max_bt)
            # pprint(alpha_pi)
            S, beta_pi = get_backwards(obs, alpha_pi)
            # pprint(beta_pi)
            fc = get_fractional_counts(alpha_pi, beta_pi, obs)
            # pprint(fc)
            accumulate_fractional_counts(fc)
            print 'accumulating', idx

    print ''
    pprint(fractional_counts)
    print 'arcs', len(conditional_arcs_to_features), 'features', len(features_to_conditional_arcs)
    pprint(features_to_conditional_arcs)










