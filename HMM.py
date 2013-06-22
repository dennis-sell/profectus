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
