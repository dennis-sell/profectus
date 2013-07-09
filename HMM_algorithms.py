import numpy

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
