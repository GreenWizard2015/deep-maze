import random
import numpy as np
import math

class CMazeExperience:
  def __init__(self, maxSize):
    self.maxSize = maxSize
    self.sizeLimit = (maxSize * 1.1)
    self.episodes = []
    self.gamma = 0.5
    self.minScore = -math.inf
  
  def addEpisode(self, replay):
    score = sum(x[2] for x in replay)
    if score < self.minScore: return
    
#     for i in range(len(replay)):
#       state, act, score, nextState = replay[i]
#       gamma = self.gamma
#       for j in range(i + 1, len(replay)):
#         score += gamma * replay[j][2]
#         gamma *= self.gamma
    self.episodes.append((replay, score))
      
    if self.sizeLimit < len(self.episodes):
      self.update()
    return

  def update(self):
    self.episodes = list(
      sorted(self.episodes, key=lambda x: x[1], reverse=True)
    )[:self.maxSize]
    self.minScore = self.episodes[-1][1]
    print('Min score: %.6f' % self.minScore)
    
  def __len__(self):
    return len(self.episodes)
  
  def take_batch(self, batch_size):
    batch = []
    weights = [x[1] for x in self.episodes]
    while len(batch) < batch_size:
      episode, _ = random.choices(
        self.episodes, 
        weights=weights, 
        k=1
      )[0]
      
      minibatchIndexes = set(random.choices(
        np.arange(len(episode)),
        weights=[abs(x[2]) for x in episode],
        k=min((5, batch_size - len(batch), len(episode)))
      ))
      
      for ind in minibatchIndexes:
        state, act, score, nextState = episode[ind]
        nextStateWeight = 1 if ind < len(episode) - 1 else 0 
        batch.append((state, act, score, nextState, nextStateWeight))

    
    return (
      np.array([x[0] for x in batch]),
      np.array([x[1] for x in batch]),
      np.array([x[2] for x in batch]),
      np.array([x[3] for x in batch]),
      np.array([x[4] for x in batch]),
    )