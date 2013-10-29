import collections
import itertools
import math
import numpy

import HMM_algorithms as Algs

class HiddenMarkovModel(object):

    def __init__(self, initial, transitions, emissions):
        """ initial - initial[x] = chance that initial state is state x. (1 x N matrix)
            transitions - transitions[x][y] = chance that state will transition to y, given that the
              model is currently at state x. (N x N matrix)
            emissions - emisions[s][o] = chance that observation o will occur given that the model
              is at state s. (M x N matrix)
            N = number of states
            M = number of observations
        """
        self.initial = numpy.matrix(initial)
        self.transitions = numpy.matrix(transitions)
        self.emissions = numpy.matrix(emissions)
        self.N = self.emissions.shape[0]   # Number of states
        self.M = self.emissions.shape[1]   # Number of obeservations
        self.states = range(self.N)
        self.observations = range(self.M)

    # Checks that
    #   1) The matrices are of proper dimensions.
    #   2) The matrices are row stochastic (probabilities add to 1)
    def is_valid(self):
        epsilon = 0.000001
        M = self.M
        N = self.N
        # Creates a list of requirements. Then checks if the list meets all of them.
        reqs = [self.initial.shape == (1,N),
                self.transitions.shape == (N,N),
                self.emissions.shape == (N,M),
                abs(self.initial.sum() - 1) < epsilon]

        transition_row_sums = self.transitions.sum(1).flatten().tolist()[0]
        reqs.extend([abs(row_sum - 1) < epsilon for row_sum in transition_row_sums])
        emission_row_sums = self.emissions.sum(1).flatten().tolist()[0]
        reqs.extend([abs(row_sum - 1) < epsilon for row_sum in emission_row_sums])
        return all(reqs)

# A class that encapsulates the hidden markov model for use with real data.
class AppliedHMM(object):
    def __init__(self, training_data, smoothing_constant=.01):
        self.hmm, self.o_mapper, self.s_mapper = parameter_estimation(training_data, smoothing_constant)

    # Determines the most likely states for a list of lists of observations.
    def decode(self, observations):
        observation_code = [self.o_mapper[o] for o in observations]
        state_code =  Algs.vitterbi(self.hmm, observation_code)
        return [self.s_mapper[s] for s in state_code]


class Mapping(dict):
    def __len__(self):
        return dict.__len__(self) / 2

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

# Used to convert real data to integers for more concise coding.
def get_mapping(list_of_sequences):
    all_elements = set(itertools.chain(*list_of_sequences))
    numbered = enumerate(list(all_elements))
    mapper = Mapping()
    for e, v in numbered:
        mapper[e] = v

    mapped_sequences = []
    for sequence in list_of_sequences:
        mapped_sequences.append([mapper[elem] for elem in sequence])
    return mapper, mapped_sequences, len(all_elements)


def stochasticize(row, smoothing_constant):
    row_total = float(sum(row))
    if row_total != 0:
        stochastic_row = [count/row_total for count in row]
    else:
        stochastic_row = row
    return stochastic_row
 

def stochasticize_matrix(matrix, smoothing_constant):
    return [stochasticize(row, smoothing_constant) for row in matrix]


# Creates a model from real data.
def parameter_estimation(training_data, smoothing_constant=.01):
    observations = [[o for o,s in row] for row in training_data]
    states = [[s for o,s in row] for row in training_data]

    state_mapper, mapped_states, num_states = get_mapping(states)
    observation_mapper, mapped_observations, num_observations = get_mapping(observations)

    # Get initial state matrix
    first_states = [sequence[0] for sequence in mapped_states]
    initial_counts = collections.Counter(first_states)
    initial = [initial_counts[s] for s in range(num_states)]
    initial = stochasticize(initial, smoothing_constant)

    # Get emissions probability matrix
    paired_states_and_observations = (zip(s_seq, o_seq) for s_seq, o_seq in zip(mapped_states, mapped_observations))
    emission_counts = collections.Counter(itertools.chain(*paired_states_and_observations))
    matrix = [ [emission_counts[(i,j)] for j in range(num_observations)] for i in range(num_states)]
    emissions = stochasticize_matrix(matrix, smoothing_constant)

    # Get state transition probability matrix
    state_transitions = itertools.chain(*[zip(sequence, sequence[1:]) for sequence in mapped_states])
    transition_counts = collections.Counter(state_transitions)
    matrix = [ [transition_counts[(i,j)] for j in range(num_states)] for i in range(num_states)]
    transitions = stochasticize_matrix(matrix, smoothing_constant)

    hmm = HiddenMarkovModel(initial, transitions, emissions)
    return hmm, observation_mapper, state_mapper

