import tensorflow as tf
import numpy as np

def train(model, memory, params):
  modelClone = tf.keras.models.clone_model(model)
  modelClone.set_weights(model.get_weights()) # use clone model for stability
  
  BOOTSTRAPPED_STEPS = params['steps']
  GAMMA = params['gamma']
  ALPHA = params.get('alpha', 1.0)
  rows = np.arange(params['batchSize'])
  lossSum = 0
  for _ in range(params['episodes']):
    allStates, actions, rewards, _, nextStateScoreMultiplier = memory.sampleSequenceBatch(
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
    
    lossSum += model.fit(states[:, 0], targets, epochs=1, verbose=0).history['loss'][0]
    ###

  return lossSum / params['episodes']
