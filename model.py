import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf

def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation="relu")(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def createModel(shape):
  inputs = res = layers.Input(shape=shape)
  res = convBlock(res, 3, filters=32)
  res = convBlock(res, 3, filters=32)
  res = convBlock(res, 3, filters=32)
  
  res = layers.Flatten()(res)
  
  # dueling dqn
  valueBranch = layers.Dense(32, activation='relu')(res)
  valueBranch = layers.Dense(32, activation='relu')(valueBranch)
  valueBranch = layers.Dense(32, activation='relu')(valueBranch)
  valueBranch = layers.Dense(1, activation='linear')(valueBranch)
  
  actionsBranch = layers.Dense(128, activation='relu')(res)
  actionsBranch = layers.Dense(64, activation='relu')(actionsBranch)
  actionsBranch = layers.Dense(64, activation='relu')(actionsBranch)
  actionsBranch = layers.Dense(64, activation='relu')(actionsBranch)
  actionsBranch = layers.Dense(4, activation='linear')(actionsBranch)
  
  res = layers.Lambda(
    lambda x: x[1] + (x[0] - tf.reduce_mean(x[0], axis=-1, keepdims=True))
  )([actionsBranch, valueBranch])

  return keras.Model(inputs=inputs, outputs=res)