import numpy
import operator

import HMM

def probability_of_observations(hmm, observations):
    """ Calculates the probability that a set of observations would occur given a hidden markov
        model. Considers all possible sequences of underlying states.
    """
    T = len(observations)
    partial_probability = []
    first_observation = observations[0]
    partial_probability = [hmm.initial[0, state] * hmm.emissions[state, first_observation]
                                                                  for state in hmm.states]
    for t in range(1,T):
        last_row = partial_probability
        o = observations[t]
        partial_probability = []
        for s2 in hmm.states:
            prob_reaching_state = sum(last_row[s1] * hmm.transitions[s1, s2] for s1 in hmm.states)
            partial_probability.append( prob_reaching_state * hmm.emissions[s2, o])

    return sum(partial_probability)


def analysis_of_state_sequences(hmm, observations):
    """ Given an hmm and a sequence of observaitions, does a brute force analysis of
        all possible state sequences to determine the most likely one.
        Used for testing vitterbi with simple models.
    """
    def all_sequences(elements, length):
        """ Generates all possible lists of a certain length with certain element"""
        possible_seqs = [[]]
        for _ in range(length):
            possible_seqs = [sequence + [n] for sequence in possible_seqs for n in elements]
        return possible_seqs

    T = len(observations)
    state_sequences = all_sequences(hmm.states, T)

    results = []
    for sequence in state_sequences:
        probability = hmm.initial[0, sequence[0]] * hmm.emissions[sequence[0], observations[0]]
        for t in range(1, T):
            probability *= hmm.transitions[sequence[t - 1], sequence[t]]
            probability *= hmm.emissions[sequence[t], observations[t]]
        results.append((sequence, probability))
    return results


def vitterbi(hmm, observations):
    """ Given a hmm, and a set of observations, determines the sequence of probabilities which
        maximize the probability of generating the observations.
     """
    T = len(observations)
    # Initialization
    first_observation = observations[0]
    delta = [hmm.initial[0,state] * hmm.emissions[state, first_observation] for state in hmm.states]
    omega = [[0] * hmm.N]
    # Recursion
    for t in range(1, T):
        old_delta, delta = delta, []
        o = observations[t]
        omega_row = []
        for s2 in hmm.states:
            values = enumerate(old_delta[s1] * hmm.transitions[s1, s2] for s1 in hmm.states)
            max_index, max_value = max(values, key=operator.itemgetter(1))
            omega_row.append(max_index)
            delta.append(max_value * hmm.emissions[s2, o])
        omega.append(omega_row)

    states = [delta.index(max(delta))]
    for t in range(1, T):
        states.insert(0, omega[t][states[0]])
    return states


if __name__ == "__main__":
    hmm = HMM.HiddenMarkovModel([.5,.5],[[.5,.5],[.5,.5]],[[.2,.8], [.8, .2]])
    for seq, prob in analysis_of_state_sequences(hmm, [0,1,1]):
        print seq, prob
