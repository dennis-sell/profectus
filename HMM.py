import math
import numpy

class HiddenMarkovModel(object):
  def __init__(initial, transitions, emmissions):
    self.initial = numpy.matrix(initial)
    self.transitions = numpy.matrix(transitions)
    self.emmisions = numpy.matrix(emmissions)
    self.N = emmisions.shape()[0]
    self.M = emmisions.shape()[1]
