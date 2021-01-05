# -*- coding: utf-8 -*-
import sys
import os
import tensorflow as tf
from Agent.MaskedSoftmax import MaskedSoftmax

if 'COLAB_GPU' in os.environ:
  # fix resolve modules
  from os.path import dirname
  sys.path.append(dirname(dirname(dirname(__file__))))
else: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3 * 1024)]
  )

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import tensorflow.keras as keras

from model import createModel
from Core.MazeRLWrapper import MazeRLWrapper
from Utils.ExperienceBuffers.CebPrioritized import CebPrioritized
from Agent.DQNAgent import DQNAgent
from Agent.DQNEnsembleAgent import DQNEnsembleAgent
import time
import Utils
from Utils.ExperienceBuffers.CebLinear import CebLinear
import glob
import numpy as np

#######################################
def train(model, trainableModel, memory, params):
  modelClone = tf.keras.models.clone_model(model)
  modelClone.set_weights(model.get_weights()) # use clone model for stability
  
  BOOTSTRAPPED_STEPS = params['steps']
  GAMMA = params['gamma']
  ALPHA = params.get('alpha', 1.0)
  rows = np.arange(params['batchSize'])
  lossSum = 0
  for _ in range(params['episodes']):
    allStates, actions, rewards, actionsMask, teacherPredictions, nextStateScoreMultiplier = memory.sampleSequenceBatch(
      batch_size=params['batchSize'],
      maxSamplesFromEpisode=params.get('maxSamplesFromEpisode', 16),
      sequenceLen=BOOTSTRAPPED_STEPS + 1
    )
    
    states = allStates[:, :-1]
    rewards = rewards[:, :-1]
    actions = actions[:, 0]

    futureScores = modelClone.predict(allStates[:, -1]).max(axis=-1) * nextStateScoreMultiplier[:, -1]
    totalRewards = (rewards * (GAMMA ** np.arange(BOOTSTRAPPED_STEPS))).sum(axis=-1)
    targets = modelClone.predict(states[:, 0])
    
    targets[rows, actions] += ALPHA * (
      totalRewards + futureScores * (GAMMA ** BOOTSTRAPPED_STEPS) - targets[rows, actions]
    )
    
    lossSum += trainableModel.fit(
      [states[:, 0], teacherPredictions[:, 0], actionsMask[:, 0], targets],
      epochs=1, verbose=0
    ).history['loss'][0]
    ###

  return lossSum / params['episodes']

def complexLoss(valueLoss, teacherPower, distributions, actionsMasks, y_true, y_pred, y_pred_softmax):
  # mask out invalid actions
  lossValues = valueLoss(y_true * actionsMasks, y_pred * actionsMasks)
  
  lossDistribution = keras.losses.kl_divergence(distributions * actionsMasks, y_pred_softmax * actionsMasks)
  return lossValues + (lossDistribution * teacherPower)

def wrapStudentModel(student):
  inputA = keras.layers.Input(shape=student.layers[0].input_shape[0][1:])
  inputDistributions = keras.layers.Input(shape=(4, ))
  inputMasks = keras.layers.Input(shape=(4, ))
  inputTargets = keras.layers.Input(shape=(4, ))
  teacherPower = tf.Variable(1.0, tf.float32)
  
  res = student(inputA)
  resSoftmax = MaskedSoftmax()(res, inputMasks)
  
  model = keras.Model(inputs=[inputA, inputDistributions, inputMasks, inputTargets], outputs=[res, resSoftmax])
  model.add_loss(complexLoss(
    Huber(delta=1),
    teacherPower,
    inputDistributions, inputMasks, inputTargets,
    res, resSoftmax
  ))
  model.compile(optimizer=Adam(lr=1e-3), loss=None )
  return model, teacherPower

def learn_environment(teacher, model, params):
  NAME = params['name']
  BATCH_SIZE = params['batch size']
  GAMMA = params['gamma']
  BOOTSTRAPPED_STEPS = params['bootstrapped steps']
  LOOP_LIMIT = params['maze']['loop limit']
  metrics = {}

  environments = [
    MazeRLWrapper(params['maze']) for _ in range(params['test episodes'])
  ]
  
  memory = CebPrioritized(maxSize=5000, sampleWeight='abs')
  doomMemory = CebLinear(
    maxSize=params.get('max steps after loop', 16) * 10000,
    sampleWeight='abs'
  )
  trainableModel, teacherPower = wrapStudentModel(model)
  ######################################################
  def withTeacherPredictions(replay):
    prevStates, actions, rewards, actionsMasks = zip(*replay)
    teacherPredictions = teacher.predict(np.array(prevStates), np.array(actionsMasks))
    return list(zip(prevStates, actions, rewards, actionsMasks, teacherPredictions))
  
  def testModel(EXPLORE_RATE):
    for e in environments: e.reset()
    replays = Utils.emulateBatch(
      environments,
      DQNAgent(model, exploreRate=EXPLORE_RATE, noise=params.get('agent noise', 0)),
      maxSteps=params.get('max test steps')
    )
    for replay, _ in replays:
      if params.get('clip replay', False):
        replay = Utils.clipReplay(replay, loopLimit=LOOP_LIMIT)
      if BOOTSTRAPPED_STEPS < len(replay):
        memory.addEpisode(withTeacherPredictions(replay), terminated=True)
    
    scores = [x.score for x in environments]
    ################
    # collect bad experience
    envs = [e for e in environments if e.hitTheLoop]
    if envs:
      for e in envs: e.Continue()
      replays = Utils.emulateBatch(
        envs,
        DQNAgent(
          model,
          exploreRate=params.get('explore rate after loop', 1),
          noise=params.get('agent noise after loop', 0)
        ),
        maxSteps=params.get('max steps after loop', 16)
      )
      for replay, _ in replays:
        if BOOTSTRAPPED_STEPS < len(replay):
          doomMemory.addEpisode(withTeacherPredictions(replay), terminated=True)
    ################
    return scores
  ######################################################
  # collect some experience
  for _ in range(2):
    testModel(EXPLORE_RATE=0)
  #######################
  bestModelScore = -float('inf')
  for epoch in range(params['epochs']):
    T = time.time()
    
    EXPLORE_RATE = params['explore rate'](epoch)
    alpha = params.get('alpha', lambda _: 1)(epoch)
    teacherP = max((0, params.get('teacher power', lambda _: 1)(epoch) ))
    teacherPower.assign(teacherP)
    print(
      '[%s] %d/%d epoch. Explore rate: %.3f. Alpha: %.5f. Teacher power: %.3f' % (
        NAME, epoch, params['epochs'], EXPLORE_RATE, alpha, teacherP
      )
    )
    ##################
    # Training
    trainLoss = train(
      model, trainableModel, memory,
      {
        'gamma': GAMMA,
        'batchSize': BATCH_SIZE,
        'steps': BOOTSTRAPPED_STEPS,
        'episodes': params['train episodes'](epoch),
        'alpha': alpha
      }
    )
    print('Avg. train loss: %.4f' % trainLoss)
    
    if BATCH_SIZE < len(doomMemory):
      trainLoss = train(
        model, trainableModel, doomMemory,
        {
          'gamma': GAMMA,
          'batchSize': BATCH_SIZE,
          'steps': BOOTSTRAPPED_STEPS,
          'episodes': params['train doom episodes'](epoch),
          'alpha': params.get('doom alpha', lambda _: alpha)(epoch)
        }
      )
    print('Avg. train doom loss: %.4f' % trainLoss)
    ##################
    # test
    print('Testing...')
    scores = testModel(EXPLORE_RATE)
    Utils.trackScores(scores, metrics)
    ##################
    
    scoreSum = sum(scores)
    print('Scores sum: %.5f' % scoreSum)
    if (bestModelScore < scoreSum) and (params['warm up epochs'] < epoch):
      print('save best model (%.2f => %.2f)' % (bestModelScore, scoreSum))
      bestModelScore = scoreSum
      model.save_weights('weights/%s.h5' % NAME)
    ##################
    os.makedirs('charts', exist_ok=True)
    Utils.plotData2file(metrics, 'charts/%s.jpg' % NAME)
    print('Epoch %d finished in %.1f sec.' % (epoch, time.time() - T))
    print('------------------')

#######################################
MAZE_FOV = 3
MAZE_MINIMAP_SIZE = 8
MAZE_LOOPLIMIT = 32
#######################################

if __name__ == "__main__":
  DEFAULT_MAZE_PARAMS = {
   'size': 40,
   'FOV': MAZE_FOV,
   'minimapSize': MAZE_MINIMAP_SIZE,
   'loop limit': MAZE_LOOPLIMIT,
  }
  
  MODEL_INPUT_SHAPE = MazeRLWrapper(DEFAULT_MAZE_PARAMS).input_size
  
  models = []
  for x in glob.iglob('weights/agent-*.h5'):
    filename = os.path.abspath(x)
    model = createModel(shape=MODEL_INPUT_SHAPE)
    model.load_weights(filename)
    models.append(model)
    
  teacher = DQNEnsembleAgent(models)
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
    'agent noise after loop': 0.1,

    'max test steps': 1000
  }
  #######################
  # just transfer distributions from teacher
  learn_environment(
    teacher,
    createModel(shape=MODEL_INPUT_SHAPE),
    {
      **DEFAULT_LEARNING_PARAMS,
      'name': 'distilled',
      'teacher power': lambda epoch: 1,
    }
  )