import numpy as np
import math
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
from Agent.MaskedSoftmax import MaskedSoftmax

def combineModels(models, combiner):
  shape = models[0].layers[0].input_shape[0][1:]
  inputs = layers.Input(shape=shape)
  actionsMask = layers.Input(shape=(4, ))
  
  predictions = [ layers.Reshape((1, -1))(
    MaskedSoftmax()( x(inputs), actionsMask )
  ) for x in models ]
  
  res = layers.Lambda(combiner)( layers.Concatenate(axis=1)(predictions)  )
  res = MaskedSoftmax()( res, actionsMask )
  return keras.Model(inputs=[inputs, actionsMask], outputs=res)

@tf.function
def multiplyOutputs(outputs):
  return tf.math.reduce_prod(1 + outputs, axis=1)

ENSEMBLE_MODE = {
  'multiply': multiplyOutputs
}

class DQNEnsembleAgent:
  def __init__(self, models, mode='multiply', exploreRate=0, noise=None):
    self._model = combineModels(models, ENSEMBLE_MODE.get(mode, mode))
    self._exploreRate = exploreRate
    self._noise = noise
    return
  
  def reset(self):
    return
  
  def process(self, state, actionsMask = [1, 1, 1, 1]):
    return self.processBatch([state], [actionsMask])[0]
    
  def processBatch(self, states, actionsMask):
    actions = self._model.predict([np.array(states), np.array(actionsMask)])
    if 0 < self._exploreRate:
      rndIndexes = np.where(np.random.random_sample(actions.shape[0]) < self._exploreRate)
      actions[rndIndexes] = np.random.random_sample(actions.shape)[rndIndexes]

    if not (self._noise is None):
      actions = normed + (np.random.random_sample(actions.shape) * self._noise)

    actions[np.where(~(1 == np.array(actionsMask)))] = -math.inf
    return actions.argmax(axis=-1)
  
  def predict(self, states, actionsMask):
    return self._model.predict([states, actionsMask])