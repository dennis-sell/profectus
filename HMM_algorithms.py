import numpy
import operator

import HMM

def probability_of_observations(hmm, observations):
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
      partial_probability.append(sum(last_row[s1] * hmm.transitions[s1, s2] for s1 in hmm.states)
                                                                           * hmm.emissions[s2, o])
  return sum(partial_probability)


def analysis_of_state_sequences(hmm, observations):

  def all_sequences(elements, length):
    possible_sequences = [[]]
    for _ in range(length):
      possible_sequences = [sequence + [n] for sequence in possible_sequences for n in elements]
    return possible_sequences

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
