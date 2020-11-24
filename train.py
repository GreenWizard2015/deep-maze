# -*- coding: utf-8 -*-
import sys
import os
import tensorflow as tf
from CMazeExperience import CMazeExperience

if 'COLAB_GPU' in os.environ:
  # fix resolve modules
  from os.path import dirname
  sys.path.append(dirname(dirname(dirname(__file__))))
else: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1 * 1024)]
  )

import random
import numpy as np

from keras.optimizers import Adam
 
from Core.CMazeEnviroment import CMazeEnviroment, MAZE_ACTIONS
from model import createModel

def emulate(env, model, exploreRate, exploreDecay, steps, stopOnInvalid=False):
  episodeReplay = []
  done = False
  N = 0
  while (N < steps) and not done:
    N += 1
    act = None
    valid = env.validActionsIndex()
    if not valid: break

    state = env.state2input()      
    if random.random() < exploreRate:
      act = random.choice(valid)
    else:
      probe = model.predict(np.array([state]))[0]
      if not stopOnInvalid:
        for i in env.invalidActions():
          probe[i] = -1
      act = np.argmax(probe)

    if stopOnInvalid and not (act in valid):
      episodeReplay.append([state, act, -1, env.state2input()])
      break
    
    prevScore = env.score
    env.apply(MAZE_ACTIONS[act])
    normedScore = 1 if 0 < (env.score - prevScore) else -.1
    episodeReplay.append([state, act, normedScore, env.state2input()])
    
    done = env.done
    exploreRate = max((.01, exploreRate * exploreDecay))
  return episodeReplay

if __name__ == "__main__":
  sz = 32
  env = CMazeEnviroment(
    maze=(0.8 < np.random.rand(sz, sz)).astype(np.float32),
    pos=(0, 0),
    FOV=3,
    minimapSize=8
  )
  memory = CMazeExperience(maxSize=100)
  done = False
  batch_size = 64
  playSteps = 64
  
  bestModelScore = 0
  model = createModel(shape=env.input_size)
  model.compile(
    optimizer=Adam(lr=1e-3),
    loss='mean_squared_error'
  )
#   model.load_weights('model.h5')
  
  targetModel = createModel(shape=env.input_size)
  np.set_printoptions(precision=3)
  # collect data
  while len(memory) < 50:
    env.respawn()
    episodeReplay = emulate(
      env, model,
      exploreRate=0.9,
      exploreDecay=0.9,
      steps=playSteps,
      stopOnInvalid=False
    ) 
    #################
    if 1 < len(episodeReplay):
      memory.addEpisode(episodeReplay)
      print(len(memory), env.score)
  memory.update()

  train_episodes = 500
  test_episodes = 10
  exploreRate = 1
  exploreDecayPerEpoch = .9
  exploreDecay = .9
  for epoch in range(5000):
    print('Epoch %d' % epoch)
    # train
    targetModel.set_weights(model.get_weights())
    lossSum = 0
    for n in range(train_episodes):
      states, actions, rewards, nextStates, nextReward = memory.take_batch(batch_size)
      targets = targetModel.predict(nextStates)
      targets[np.arange(len(targets)), actions] = rewards + np.max(targets, axis=1) * .9 * nextReward

      lossSum += model.fit(
        states, targets,
        epochs=1,
        verbose=0
      ).history['loss'][0]
    print('Avg. train loss: %.4f' % (lossSum / train_episodes))
    print(targets[0])

    # test
    print('Epoch %d testing' % epoch)
    bestScore = scoreSum = movesSum = 0
    n = 0
    while n < test_episodes:
      env.respawn()
      episodeReplay = emulate(
        env, model,
        exploreRate=exploreRate,
        exploreDecay=exploreDecay,
        steps=playSteps*2,
        stopOnInvalid=True
      )
      if 1 < len(episodeReplay):
        memory.addEpisode(episodeReplay)
        n += 1
        bestScore = max((bestScore, env.score))
        scoreSum += env.score
        movesSum += len(episodeReplay)
      #################
    print('Best score: %.3f, avg. score: %.3f, avg. moves: %.1f' % (bestScore, scoreSum / n, movesSum / n))
    if bestModelScore < scoreSum:
      bestModelScore = scoreSum
      print('save best model')
      model.save_weights('model.h5')
    model.save_weights('latest.h5')
    exploreRate *= exploreDecayPerEpoch