import numpy as np
import math

class DQNAgent:
  def __init__(self, model, exploreRate=0, noise=None):
    self._model = model
    self._exploreRate = exploreRate
    self._noise = noise
    return
  
  def reset(self):
    return
  
  def process(self, state, actionsMask = [1, 1, 1, 1]):
    return self.processBatch([state], [actionsMask])[0]
    
  def processBatch(self, states, actionsMask):
    actions = self._model.predict(np.array(states))
    if 0 < self._exploreRate:
      rndIndexes = np.where(np.random.random_sample(actions.shape[0]) < self._exploreRate)
      actions[rndIndexes] = np.random.random_sample(actions.shape)[rndIndexes]

    if not (self._noise is None):
      # softmax
      e_x = np.exp(actions - actions.max(axis=-1, keepdims=True))
      normed = e_x / e_x.sum(axis=-1, keepdims=True)
      # add noise
      actions = normed + (np.random.random_sample(actions.shape) * self._noise)

    actions[np.where(~(1 == np.array(actionsMask)))] = -math.inf
    return actions.argmax(axis=-1)