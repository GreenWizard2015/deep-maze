# -*- coding: utf-8 -*-
import sys
import os
import tensorflow as tf

if 'COLAB_GPU' in os.environ:
  # fix resolve modules
  from os.path import dirname
  sys.path.append(dirname(dirname(dirname(__file__))))
else: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3 * 1024)]
  )

from learn_environment import learn_environment
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from model import createModel
from Core.MazeRLWrapper import MazeRLWrapper

#######################################
MAZE_FOV = 3
MAZE_MINIMAP_SIZE = 8
MAZE_LOOPLIMIT = 32
#######################################

def getModel(shape):
  model = createModel(shape=MODEL_INPUT_SHAPE)
  model.compile(optimizer=Adam(lr=1e-3), loss=Huber(delta=1))
  return model

if __name__ == "__main__":
  DEFAULT_MAZE_PARAMS = {
   'size': 40,
   'FOV': MAZE_FOV,
   'minimapSize': MAZE_MINIMAP_SIZE,
   'loop limit': MAZE_LOOPLIMIT,
  }
  
  MODEL_INPUT_SHAPE = MazeRLWrapper(DEFAULT_MAZE_PARAMS).input_size
  
  #######################
  DEFAULT_LEARNING_PARAMS = {
    'maze': DEFAULT_MAZE_PARAMS,
    'batch size': 256,
    'gamma': 0.95,
    'bootstrapped steps': 3,
    
    'epochs': 100,
    'warm up epochs': 0,
    'test episodes': 128,
    'train episodes': lambda _: 128,
    'train doom episodes': lambda _: 32,

    'alpha': lambda _: 1,
    'explore rate': lambda _: 0,
    
    'agent noise': 0.01,
    'clip replay': True,
    
    'explore rate after loop': 0.2,
    'agent noise after loop': 0.1
  }
  #######################
  for i in range(4):
    learn_environment(
      getModel(MODEL_INPUT_SHAPE),
      {
        **DEFAULT_LEARNING_PARAMS,
        'name': 'agent-%d' % i,
        'max test steps': 1000
      }
    )