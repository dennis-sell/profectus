import math
import numpy

epsilon = 0.000001

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

  def is_valid(self):
    M = self.M
    N = self.N
    # Creates a list of requirements. Then checks if the list meets all of them.
    reqs = [self.initial.shape == (1,N),
            self.transitions.shape == (N,N),
            self.emissions.shape == (M,N),
            abs(self.initial.sum() - 1) < epsilon]

    transition_row_sums = self.transitions.sum(1).flatten().tolist()[0]
    reqs.extend([abs(row_sum - 1) < epsilon for row_sum in transition_row_sums])
    emission_row_sums = self.emissions.sum(1).flatten().tolist()[0]
    reqs.extend([abs(row_sum - 1) < epsilon for row_sum in emission_row_sums])
    return reqs.count(False) > 0
