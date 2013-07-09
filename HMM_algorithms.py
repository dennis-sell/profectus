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


def probability_of_observations_by_state_sequence(hmm, observations): 
  possible_state_sequences = [[state] for state in hmm.states]
  for i in range(len(observations) - 1):
    possible_state_sequences = [sequence + [next] for sequence in possible_state_sequences for next in hmm.states]
  results = []
  for sequence in possible_state_sequences: 
    current_state = sequence[0]
    probability = hmm.initial[0, current_state] * hmm.emissions[current_state, observations[0]]
    for t in range(1, T):
      probability *= hmm.transitions[sequence[t - 1], sequence[t]] * hmm.emissions[sequence[t], observations[t]]
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
    old_delta = delta
    o = observations[t]
    delta = []
    omega_row = []
    for s2 in hmm.states:
      max_index, max_value = max(enumerate(old_delta[s1] * hmm.transitions[s1, s2] for s1 in hmm.states), key=operator.itemgetter(1))
      omega_row.append(max_index)
      delta.append(max_value * hmm.emissions[s2, o])  
    omega.append(omega_row)
  
  states = [delta.index(max(delta))]
  for t in range(1, T):
    states.insert(0, omega[t][states[0]])
  return state

if __name__ == "__main__":
  probability_of_observations_by_state_sequence(HMM.HiddenMarkovModel([.5],[[.5,.5],[.5,.5]],[[.2,.8], [.8, .2]]), [0,1,1])
