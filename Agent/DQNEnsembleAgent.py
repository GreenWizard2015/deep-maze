import numpy as np
import math
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf

def combineModels(models, combiner):
  shape = models[0].layers[0].input_shape[0][1:]
  inputs = layers.Input(shape=shape)
  actionsMask = layers.Input(shape=(4, ))
  res = layers.Lambda(combiner)([actionsMask] + [ x(inputs) for x in models ])
  return keras.Model(inputs=[inputs, actionsMask], outputs=res)

def maskedSoftmax(mask, inputs):
  mask = tf.where(tf.equal(mask, 1))
  return [
    tf.sparse.to_dense(
      tf.sparse.softmax(
        tf.sparse.SparseTensor(
          indices=mask,
          values=tf.gather_nd(x, mask),
          dense_shape=tf.shape(x, out_type=tf.int64)
        )
      )
    ) for x in inputs
  ]
  
def multiplyOutputs(inputs):
  outputs = maskedSoftmax(inputs[0], inputs[1:])
  
  res = 1 + outputs[0]
  for x in outputs[1:]:
    res = tf.math.multiply(res, 1 + x)
  return res

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
      # softmax
      e_x = np.exp(actions - actions.max(axis=-1, keepdims=True))
      normed = e_x / e_x.sum(axis=-1, keepdims=True)
      # add noise
      actions = normed + (np.random.random_sample(actions.shape) * self._noise)

    actions[np.where(~(1 == np.array(actionsMask)))] = -math.inf
    return actions.argmax(axis=-1)