# -*- coding: utf-8 -*-
import sys
import os
import tensorflow as tf
import Utils

if 'COLAB_GPU' in os.environ:
  # fix resolve modules
  from os.path import dirname
  sys.path.append(dirname(dirname(dirname(__file__))))
else: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3 * 1024)]
  )

from model import createModel
from Core.MazeRLWrapper import MazeRLWrapper
import glob
from Agent.DQNAgent import DQNAgent
from Agent.DQNEnsembleAgent import DQNEnsembleAgent
import pylab as plt
import numpy as np

#######################################
MAZE_FOV = 3
MAZE_MINIMAP_SIZE = 8
#######################################

def plot2file(data, filename, chartname):
  plt.clf()

  figSize = plt.rcParams['figure.figsize']
  fig = plt.figure(figsize=(figSize[0] * 2, figSize[1]))
  
  axe = fig.subplots()
  series = data[chartname]
  for name, dataset in series.items():
    axe.plot(dataset, label=name)
  axe.title.set_text(chartname)
    
  fig.legend()
  fig.savefig(filename)
  plt.close(fig)
  return

def testAgent(environments, agent, name, metrics, N=5):
  print('Agent: %s' % name)
  
  scoreTop90 = metrics['scores']['%s worst 10%%' % name] = []
  
  for i in range(N):
    print('Round %d/%d...' % (i, N))
    scores = []
    
    for e in environments: e.reset()
    replays = Utils.emulateBatch(environments, agent, maxSteps=1000)
    for (replay, _), env in zip(replays, environments):
      scores.append(env.score)
    
    scores = list(sorted(scores, reverse=True))
    scoreTop90.append(scores[int(0.9 * len(scores))])

    plot2file(metrics, 'chart.jpg', 'scores')
  return

if __name__ == "__main__":
  MAZE_PARAMS = {
    'size': 64,
    'FOV': MAZE_FOV,
    'minimapSize': MAZE_MINIMAP_SIZE,
    'loop limit': 64,
  }
  environments = [MazeRLWrapper(MAZE_PARAMS) for _ in range(100)]
  MODEL_INPUT_SHAPE = environments[0].input_size

  metrics = {
    'scores': {}
  }
  models = []
  for i, x in enumerate(glob.iglob('weights/*.h5')):
    filename = os.path.abspath(x)
    model = createModel(shape=MODEL_INPUT_SHAPE)
    model.load_weights(filename)
    models.append(model)
    
    testAgent(
      environments,
      DQNAgent(model),
      name=os.path.basename(filename)[:-3],
      metrics=metrics
    )

  testAgent(
    environments,
    DQNEnsembleAgent(models),
    name='ensemble',
    metrics=metrics
  )