import HMM
import unittest

class TestSequenceFunctions(unittest.TestCase):

  def setUp(self):
    initial = [.5,.5]
    transitions = [[.7,.3], [.6,.4]]
    emmisions = [[0, .3, .7], [.5, .3, .2]]
    self.model1 = HMM.HiddenMarkovModel(initial, transitions, emmisions)
    
    emmisions = transitions
    self.model2 = HMM.HiddenMarkovModel(initial, transitions, emmisions) 

  def test_is_valid(self):
    self.assertTrue(self.model1.is_valid())
    self.assertFalse(self.model2.is_valid())

    

if __name__ == "__main__":
  unittest.main()
