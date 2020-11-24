import tensorflow.keras as keras
import tensorflow.keras.layers as layers

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

  res = layers.Dense(16 ** 2, activation='relu')(res)
  res = layers.Dropout(.2)(res)
  res = layers.Dense(16 ** 2, activation='relu')(res)
  res = layers.Dropout(.2)(res)
  res = layers.Dense(16 ** 2, activation='relu')(res)
  res = layers.Dropout(.2)(res)
  res = layers.Dense(8 ** 2, activation='relu')(res)
  res = layers.Dropout(.2)(res)
  res = layers.Dense(8 ** 2, activation='relu')(res)
  res = layers.Dropout(.2)(res)
  res = layers.Dense(8 ** 2, activation='relu')(res)
  res = layers.Dropout(.2)(res)
  res = layers.Dense(4 ** 2, activation='relu')(res)
  res = layers.Dropout(.2)(res)
  res = layers.Dense(4 ** 2, activation='relu')(res)
  res = layers.Dropout(.2)(res)
  res = layers.Dense(4 ** 2, activation='relu')(res)
  res = layers.Dropout(.2)(res)
  
  res = layers.Dense(4, activation='linear')(res)
  return keras.Model(
    inputs=inputs,
    outputs=res
  )