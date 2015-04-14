import collections
import itertools
import numpy
import operator

class HiddenMarkovModel(object):

    def __init__(self, initial, transitions, emissions):
        """ initial[x] = chance that initial state is state x. (1 x N matrix)
            transitions[x, y] = chance that state x will transition to y (N x N matrix)
            emisions[s, o] = chance that observation o will occur given that the model
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

    def vitterbi(self, observations):
        """ Given a set of observations, determines the sequence of states
            which make the observations most likely.
         """
        T = len(observations)
        # Initialization
        first_observation = observations[0]
        delta = [self.initial[0,state] * self.emissions[state, first_observation] for state in self.states]
        omega = [[0] * self.N]
        # Recursion
        for t in range(1, T):
            old_delta, delta = delta, []
            o = observations[t]
            omega_row = []
            for s2 in self.states:
                values = enumerate(old_delta[s1] * self.transitions[s1, s2] for s1 in self.states)
                max_index, max_value = max(values, key=operator.itemgetter(1))
                omega_row.append(max_index)
                delta.append(max_value * self.emissions[s2, o])
            omega.append(omega_row)

        states = [delta.index(max(delta))]
        for t in range(1, T):
            states.insert(0, omega[T-t][states[0]])

        return states


# A class that encapsulates the hidden markov model for use with real data.
class AppliedHMM(object):
    def __init__(self, training_data, smoothing_constant=.01):
        self.hmm, self.o_mapper, self.s_mapper = \
            parameter_estimation(training_data, smoothing_constant)

    def can_decode(self, observations):
        return all(o in self.o_mapper for o in observations)

    # Determines the most likely states for a list of lists of observations.
    def decode(self, observations):
        observation_code = [self.o_mapper[o] for o in observations]
        state_code =  self.hmm.vitterbi(observation_code)
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


def parameter_estimation(training_data, smoothing_constant=.01):
    """ Creates a model from real data. """
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
