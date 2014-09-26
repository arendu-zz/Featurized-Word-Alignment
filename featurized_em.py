__author__ = 'arenduchintala'

from optparse import OptionParser
from math import exp, log
import utils
from pprint import pprint

global BOUNDARY_STATE, END_STATE, SPLIT, mle_count, mle_probs, E_TYPE, T_TYPE, theta, possible_states
BOUNDARY_STATE = "###"
SPLIT = "###/###"
E_TYPE = "emission"
E_TYPE_PRE = "prefix_feature"
E_TYPE_SUF = "suffix_feature"
T_TYPE = "transition"
S_TYPE = "state"
ALL = "***"
mle_count = {}
probs = {}
theta = {}
possible_states = set([])
features_all = {}


def populate_features(mle):
    for type, f1, f2 in mle:
        # f1 is the decision variable
        # f2 is the context variable
        n = features_all.get((type, f2), set([]))
        n.add(f1)
        features_all[type, f2] = n


def extract_features(type, state=None, prev_state=None, obs=None, prev_obs=None):
    if type == E_TYPE:
        # suffix feature, prefix feature
        return [(E_TYPE_SUF, obs[-4:], state), (E_TYPE_PRE, obs[:4], state), (E_TYPE, obs, state)]
    elif type == T_TYPE:
        return [(T_TYPE, state, prev_state)]


def get_emission(obs, state):
    fired_features = extract_features(type=E_TYPE, state=state, obs=obs)
    theta_dot_features = sum([theta.get(f, 0.0) for f in fired_features])
    normalizing_decisions = features_all[E_TYPE, state]
    theta_dot_normalizing_features = 0.0
    for d in normalizing_decisions:
        d_features = extract_features(type=E_TYPE, state=state, obs=d)
        theta_dot_normalizing_features += exp(sum([theta.get(f, 0.0) for f in d_features]))
    log_prob = theta_dot_features - theta_dot_normalizing_features
    return log_prob


def get_transition(state, prev_state):
    fired_features = extract_features(type=T_TYPE, state=state, prev_state=prev_state)
    theta_dot_features = sum([theta.get(f, 0.0) for f in fired_features])
    normalizing_decisions = features_all[T_TYPE, prev_state]
    theta_dot_normalizing_features = 0.0
    for d in normalizing_decisions:
        d_features = extract_features(type=T_TYPE, state=d, prev_state=prev_state)
        theta_dot_normalizing_features += exp(sum([theta.get(f, 0.0) for f in d_features]))
    log_prob = theta_dot_features - theta_dot_normalizing_features
    return log_prob


def get_possible_states(o):
    if o == BOUNDARY_STATE:
        return [BOUNDARY_STATE]
    else:
        # TODO: this can be reduced to only co-occuring states
        return list(possible_states)


def get_backwards(raw_words, alpha_pi):
    words = [BOUNDARY_STATE] + raw_words + [BOUNDARY_STATE]
    n = len(words) - 1  # index of last word
    beta_pi = {(n, BOUNDARY_STATE): 0.0}
    posterior_unigrams = {}
    posterior_obs_accumilation = {}
    posterior_bigrams_accumilation = {}
    S = alpha_pi[(n, BOUNDARY_STATE)]  # from line 13 in pseudo code
    for k in range(n, 0, -1):
        for v in get_possible_states(words[k]):
            e = get_emission(words[k], v)
            pb = beta_pi[(k, v)]
            posterior_unigram_val = beta_pi[(k, v)] + alpha_pi[(k, v)] - S
            # posterior_obs_accumilation = do_accumilate_posterior_obs(posterior_obs_accumilation, words[k], v, posterior_unigram_val)
            # posterior_unigrams = do_append_posterior_unigrams(posterior_unigrams, k, v, posterior_unigram_val)

            for u in get_possible_states(words[k - 1]):
                # print 'reverse transition', 'k', k, 'u', u, '->', 'v', v
                q = get_transition(v, u)
                p = q + e
                beta_p = pb + p
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = utils.logadd(beta_pi[new_pi_key], beta_p)
                    # print 'beta     ', new_pi_key, '=', beta_pi[new_pi_key], exp(beta_pi[new_pi_key])
                posterior_bigram_val = alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S
                # posterior_bigram_val = "%.3f" % (exp(alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S))
                # posterior_bigrams_accumilation = do_accumilate_posterior_bigrams(posterior_bigrams_accumilation, v, u, posterior_bigram_val)
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


if __name__ == "__main__":
    opt = OptionParser()
    opt.add_option("-i", dest="initial_train", default="data/entrain4k")
    opt.add_option("-t", dest="raw", default="data/enraw")
    (options, _) = opt.parse_args()
    for t in open(options.initial_train, 'r').read().split(SPLIT):
        if t.strip() != '':
            obs_state = [tuple(x.split('/')) for x in t.split('\n') if x.strip() != '']
            obs_state.append((BOUNDARY_STATE, BOUNDARY_STATE))
            obs_state.insert(0, (BOUNDARY_STATE, BOUNDARY_STATE))
            for idx, (obs, state) in enumerate(obs_state[1:]):
                possible_states.add(state)
                prev_obs, prev_state = obs_state[idx - 1]
                mle_count[T_TYPE, state, prev_state] = mle_count.get((T_TYPE, state, prev_state), 0) + 1.0
                mle_count[E_TYPE, obs, state] = mle_count.get((E_TYPE, obs, state), 0) + 1.0
                mle_count[S_TYPE, prev_state, None] = mle_count.get((S_TYPE, prev_state), 0) + 1.0
    populate_features(mle_count)
    possible_states.remove(BOUNDARY_STATE)
    print possible_states
    s = "this is a good sentence".split()
    max_bt, max_p, alpha_pi = get_viterbi_and_forward(s)
    print(max_bt)
    pprint(alpha_pi)
    S, beta_pi = get_backwards(s, alpha_pi)
    pprint(beta_pi)










