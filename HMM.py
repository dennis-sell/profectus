import math
import numpy

class HiddenMarkovModel(object):
  

  def __init__(initial, transitions, emmissions):
    """ initial - initial[x] = chance that initial state is state x. (1 x N matrix)
        transitions - transitions[x][y] = chance that state will transition to y, given that the
          model is currently at state x. (N x N matrix)
        emmissions - emmisions[s][o] = chance that observation o will occur given that the model
          is at state s. (M x N matrix)
        N = number of states
        M = number of observations
    """
    self.initial = numpy.matrix(initial)
    self.transitions = numpy.matrix(transitions)
    self.emmisions = numpy.matrix(emmissions)
    self.N = emmisions.shape()[0]   # Number of states
    self.M = emmisions.shape()[1]   # Number of obeservations

    def is_valid(self):
      M = self.M
      N = self.N
      epsilon = 0.000001
      # Creates a list of requirements. Then checks if the list meets all of them.
      reqs = [initial.shape == (1,N),
              transitions.shape == (N,N),
              emmissions.shape == (M,N),
              abs(initial.sum() - 1) < epsilon]

      transition_row_sums = transitions.sum(1).flatten().tolist()[0]
      reqs.extend([abs(row_sum - 1) < epsilon for row_sum in transition_row_sums])
      emmision_row_sums = emmisions.sum(1).flatten().tolist()[0]
      reqs.extend([abs(row_sum - 1) < epsilon for row_sum in emmision_row_sums])
      return reqs.count(False) > 0

