from Core.CMazeEnvironment import CMazeEnvironment, MAZE_ACTIONS
import numpy as np
import math

class MazeRLWrapper:
  def __init__(self, params):
    maze = (
      params.get('obstacles rate', 0.8) < np.random.rand(params['size'], params['size'])
    ).astype(np.float32)
    
    env = CMazeEnvironment(
      maze=maze,
      pos=(0, 0),
      FOV=params['FOV'],
      minimapSize=params['minimapSize']
    )
    env.respawn()
    self._env = env
    
    self._stepsLimit = params['loop limit']
    self._minUniqSteps = params.get('min unique positions rate', 0.3)
    self._stopIfLoop = params.get('stop if loop', True)
    self._onlyNewCells = params.get('only new cells reward', False)
    return
  
  def reset(self):
    self._stopInLoop = False
    self._done = False
    self._env.respawn()
    self._moves = []
    return
    
  def apply(self, actionIndex):
    act = MAZE_ACTIONS[actionIndex]
    prevState = self.state
    prevScore = self.score
    isNewCell = not self._env.isMovingToVisited(act)
    self._env.apply(act)
    nextState = self.state
    
    self._done = True
    if self._env.dead: # unreachable due to actions masking 
      return nextState, -10, True, prevState

    if 0.99 <= self._env.score: 
      return nextState, 0, True, prevState
    
    if self._movingLoop():
      return nextState, 0, True, prevState

    self._done = False
    reward = 0.3 if isNewCell else 0 # small reward for visiting new cell
    
    if not self._onlyNewCells:
      discovered = (self._env.score - prevScore) / self._env.minScoreDelta
      reward += 1 + math.log(discovered, 10) if 0 < discovered else -1
    return nextState, reward, False, prevState
  
  def actionsMask(self):
    return self._env.actionsMask()
  
  @property
  def state(self):
    return self._env.state2input()
  
  @property
  def done(self):
    return self._done
  
  @property
  def hitTheLoop(self):
    return self._stopInLoop
  
  @property
  def score(self):
    return self._env.score
  
  @property
  def input_size(self):
    return self._env.input_size
  
  @property
  def uniqueMoves(self):
    if self._stepsLimit <= len(self._moves):
      return len(set(self._moves)) / len(self._moves)
    return 1
  
  def _movingLoop(self):
    self._moves.append(str(self._env.pos))
    self._moves = self._moves[1:] if self._stepsLimit < len(self._moves) else self._moves
    self._stopInLoop = self._stopIfLoop and (self.uniqueMoves < self._minUniqSteps)
    return self._stopInLoop
  
  def Continue(self):
    self._done = False
    self._moves = []
    return