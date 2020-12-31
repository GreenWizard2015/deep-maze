import numpy as np
import math

class RandomAgent:
  def __init__(self):
    return
  
  def reset(self):
    pass

  def process(self, state, actionsMask = [1, 1, 1, 1]):
    return self.processBatch([state], [actionsMask])
  
  def processBatch(self, states, actionsMask):
    actions = np.random.random_sample((np.array(states).shape[0], 4))
    actions[np.where(~(1 == np.array(actionsMask)))] = -math.inf
    return actions.argmax(axis=-1)