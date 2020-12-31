import random
import numpy as np
import math
import itertools

_WEIGHTS_MODES = {
  'abs': math.fabs,
  'reward': lambda x: x,
  'same': lambda _: 1
}

class CebPrioritized:
  def __init__(self, maxSize, sampleWeight='abs'):
    self.maxSize = maxSize
    self.sizeLimit = math.floor(maxSize * 1.1)
    self.episodes = []
    self.minScore = -math.inf
    self._sampleWeight = _WEIGHTS_MODES.get(sampleWeight, sampleWeight)
  
  def addEpisode(self, replay, terminated):
    score = sum(x[2] for x in replay) # state, action, 2 - reward
    if score < self.minScore: return
    self.episodes.append((replay, score, terminated))

    if self.sizeLimit < len(self.episodes):
      self.update()
    return

  def update(self):
    self.episodes = list(
      sorted(self.episodes, key=lambda x: x[1], reverse=True)
    )[:self.maxSize]
    self.minScore = self.episodes[-1][1]
    return 
    
  def __len__(self):
    return len(self.episodes)
  
  def _sampleIndexes(self, episode, maxSamples):
    return set(random.choices(
      np.arange(len(episode)),
      weights=[self._sampleWeight(x[2]) for x in episode],
      k=min((maxSamples, len(episode)))
    ))
  
  def _createBatch(self, batch_size, sampler):
    batchSize = 0
    cumweights = list(itertools.accumulate(x[1] for x in self.episodes))
    res = []
    while batchSize < batch_size:
      Episode = random.choices(self.episodes, cum_weights=cumweights, k=1)[0]
      for sample, rewardMultiplier in sampler(Episode, batch_size - batchSize):
        while len(res) <= len(sample):  res.append([])
        for i, value in enumerate(sample):
          res[i].append(value)
        res[-1].append(rewardMultiplier)
        batchSize += 1

    return [np.array(values) for values in res]
    
  def sampleBatch(self, batch_size, maxSamplesFromEpisode=5):
    def sampler(Episode, limit):
      limit = min((maxSamplesFromEpisode, limit))
      episode, _, wasTerminated = Episode
      lastActionScore = 1 if wasTerminated else 0
      minibatchIndexes = self._sampleIndexes(episode, limit)
      for ind in minibatchIndexes:
        yield ((
          episode[ind],
          lastActionScore if ind == len(episode) - 1 else 1 # last action in replay?
        ))
      return
        
    return self._createBatch(batch_size, sampler)

  def _sampleEpisodeMultipleSteps(self, Episode, maxSamplesFromEpisode, steps):
    episode, _, wasTerminated = Episode
    lastActionScore = 1 if wasTerminated else 0
    minibatchIndexes = self._sampleIndexes(
      episode[:-(steps - 1)] if 1 < steps else episode,
      maxSamplesFromEpisode
    )
    for ind in minibatchIndexes:
      sample = episode[ind:ind+steps]
      transposed = [
        np.array([x[col] for x in sample]) for col in range(len(sample[0]))
      ]
      yield((
        transposed,
        np.array([ # last action in replay?
          (lastActionScore if (ind + i) == len(episode) - 1 else 1) for i in range(steps)
        ])
      ))
    return
  
  def sampleSequenceBatch(self, batch_size, sequenceLen, maxSamplesFromEpisode=5):
    def sampler(Episode, limit):
      return self._sampleEpisodeMultipleSteps(
        Episode,
        maxSamplesFromEpisode=min((maxSamplesFromEpisode, limit)),
        steps=sequenceLen
      )
    
    return self._createBatch(batch_size, sampler)