import HMM
import unittest
import HMM_algorithms

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

  def test_is_valid(self):
    self.assertTrue(self.model1.is_valid())
    self.assertTrue(self.model2.is_valid())

  def test_calculate_probability(self):
    observations = [0,2,1]
    self.assertAlmostEquals(HMM_algorithms.probability_of_observations(self.model1, observations),  .0375)

    

if __name__ == "__main__":
  unittest.main()
