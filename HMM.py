import collections
import itertools
import math
import numpy

EPSILON = 0.000001

class Mapping(dict):
    def __len__(self):
        return dict.__len__(self) / 2

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

Model = collections.namedtuple('Model', ['model', 'o_mapper', 's_mapper'])

def get_mapping(list_of_sequences):
  all_elements = set(itertools.chain(*list_of_sequences))
  numbered = list(enumerate(list(all_elements)))
  mapper = Mapping()
  for e, v in numbered:
    mapper[e] = v

  mapped_sequences = []
  for sequence in list_of_sequences:
    mapped_sequences.append([mapper[elem] for elem in sequence])
  return mapper, mapped_sequences

def ParameterEstimation(observations, states):
  # Checks if inputs have the same size.
  if len(observations) != len(states) or not observations:
    raise ValueError("Invalid number of states and observation sequences")
  if not all(len(o) == len(s) and o for o,s in zip(observations, states)):
    raise  ValueError("Observations and states do not match in size.")

  total_sequences = len(states)
  total_instances = len(itertools.chain(*states))

  state_mapper, mapped_states = get_mapping(states)
  observation_mapper, mapped_observations = get_mapping(observations)

  first_states = [sequence[0] for sequence in states]
  initial_counts = collections.Counter(first_states)
  num_states = len(Mapping)
  initial = [float(initial_counts[s])/total_sequences for s in range(num_states)]

  paired_states_and_observations = (zip(s_seq, o_seq) for s_seq, o_seq in zip(states, observations))
  emission_counts = Counter(itertools.chain(*paired_states_and_observations))
  matrix = [ [emission_counts[(i,j)] for i in range(num_states)] for j in range(num_states)]
  emissions = []
  for row in matrix:
    row_total = float(sum(row))
    stochastic_row = [count/row_total for count in row]
    emissions.append(stochastic_row)

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

  def is_valid(self):
    M = self.M
    N = self.N
    # Creates a list of requirements. Then checks if the list meets all of them.
    reqs = [self.initial.shape == (1,N),
            self.transitions.shape == (N,N),
            self.emissions.shape == (N,M),
            abs(self.initial.sum() - 1) < EPSILON]

    transition_row_sums = self.transitions.sum(1).flatten().tolist()[0]
    reqs.extend([abs(row_sum - 1) < EPSILON for row_sum in transition_row_sums])
    emission_row_sums = self.emissions.sum(1).flatten().tolist()[0]
    reqs.extend([abs(row_sum - 1) < EPSILON for row_sum in emission_row_sums])
    return all(reqs)

if __name__ == "__main__":
  ParameterEstimation([["a","b","c"]], [["str1", "str2", "str3"]])
