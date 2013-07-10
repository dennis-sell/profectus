import unittest

import HMM
import HMM_algorithms as Algs

class TestSequenceFunctions(unittest.TestCase):

  def setUp(self):
    initial = [.5,.5]
    transitions = [[.7, .3],
                   [.6, .4]]
    emissions = [[.0, .3, .7],
                 [.5, .3, .2]]
    self.model1 = HMM.HiddenMarkovModel(initial, transitions, emissions)

    emissions = transitions
    self.model2 = HMM.HiddenMarkovModel(initial, transitions, emissions)

    self.model3 = HMM.HiddenMarkovModel([.5,.5],[[.5,.5],[.5,.5]],[[.2,.8], [.8, .2]])

  def test_is_valid(self):
    self.assertTrue(self.model1.is_valid())

  def test_is_valid2(self):
    self.assertTrue(self.model2.is_valid())

  def test_calculate_probability(self):
    observations = [0,2,1]
    self.assertAlmostEquals(Algs.probability_of_observations(self.model1, observations),  .0375)

  def test_calculate_probability2(self):
    observations = [1,1]
    self.assertAlmostEquals(Algs.probability_of_observations(self.model2, observations), .1175)


  def test_probability_by_state_sequence(self):
    """ Sum of the probabilities of all state sequences with a set of observations
        should be equal to probability that observations occur.
    """
    observations = [0,1,1]
    probabilities = Algs.analysis_of_state_sequences(self.model3, observations)
    total_probability = sum(prob for sequence, prob in probabilities)
    self.assertAlmostEquals(total_probability, Algs.probability_of_observations(self.model3, observations))

if __name__ == "__main__":
  unittest.main()
