from enum import Enum
import numpy as np
import random

class MazeActions(Enum):
  LEFT = (-1, 0)
  RIGHT = (1, 0)
  UP = (0, -1)
  DOWN = (0, 1)
  
MAZE_ACTIONS_AS_INT = { x: i for i, x in enumerate(MazeActions) }
MAZE_ACTIONS = [x for x in MazeActions]

class CMazeEnvironment:
  def __init__(self, maze, pos, FOV, minimapSize):
    self.maze = np.pad(np.array(maze), FOV, constant_values=(1,))
    self.minimapSize = minimapSize
    self._fov = self.FOV = FOV
    
    x, y = np.array(pos) + FOV
    self.spawnAt(x, y)
    return

  def spawnAt(self, x, y):
    self._steps = 0
    self.pos = np.array([y, x])
    self.fog = np.zeros_like(self.maze)
    self.moves = np.zeros_like(self.maze)
    self._update()
    return
  
  def respawn(self):
    w, h = self.maze.shape
    while True:
      x = random.randint(0, w - 1)
      y = random.randint(0, h - 1)
      if self.maze[x, y] <= 0:
        self.spawnAt(x, y)
        break
    return
  
  def _update(self):
    y, x = self.pos
    d = self._fov
    self.fog[x - d:x + d + 1, y - d:y + d + 1] = 1
    self.moves *= .98
    self.moves[x, y] = 1
    return
  
  def apply(self, action):
    self._steps += 1
    self.pos += action.value
    self._update()
    return
  
  def isMovingToVisited(self, action):
    y, x = self.pos + action.value
    return 1 == self.moves[x, y]

  def vision(self):
    y, x = self.pos
    return self.maze[
      x - self._fov:x + self._fov + 1,
      y - self._fov:y + self._fov + 1
    ]

  def _takeShot(self):
    maze, fog, moves = self.maze, self.fog, self.moves
    y, x = self.pos
    h, w = self.maze.shape
    
    isXAxisOk = (self.minimapSize < x) and (x < (w - self.minimapSize))
    isYAxisOk = (self.minimapSize < y) and (y < (h - self.minimapSize))
    if not (isXAxisOk and isYAxisOk):
      x += self.minimapSize
      y += self.minimapSize
      maze = np.pad(maze, self.minimapSize, constant_values=(1,))
      fog, moves = (
        np.pad(data, self.minimapSize, constant_values=(0,)) for data in (fog, moves)
      )

    d = self.minimapSize
    return (data[x - d:x + d + 1, y - d:y + d + 1] for data in (maze, fog, moves))
  
  def minimap(self):
    maze, fog, moves = self._takeShot()
    return ((maze * fog) - (1 - fog), moves)
  
  @property
  def dead(self):
    y, x = self.pos
    return 0 < self.maze[x, y]
  
  @property
  def score(self):
    return np.count_nonzero(self.fog) * self.minScoreDelta
  
  @property
  def steps(self):
    return self._steps
  
  def copy(self):
    # dirty copy
    res = CMazeEnvironment(self.maze, self.pos, self._fov, self.minimapSize)
    res.maze = self.maze.copy()
    res.fog = self.fog.copy()
    res.pos = self.pos.copy()
    res.moves = self.moves.copy()
    return res
  
  def isPossible(self, action):
    y, x = self.pos + action.value
    return self.maze[x, y] <= 0
  
  def validActions(self):
    return [ act for act in MazeActions if self.isPossible(act) ]
  
  def validActionsIndex(self):
    return [ i for i, act in enumerate(MazeActions) if self.isPossible(act) ]
  
  def actionsMask(self):
    return [ (1 if self.isPossible(act) else 0) for act in MazeActions ]
  
  def invalidActions(self):
    return [ i for i, act in enumerate(MazeActions) if not self.isPossible(act) ]
  
  def state2input(self):
    maze, moves = self.minimap()
    state = np.dstack((maze, moves))
    return state

  @property
  def input_size(self):
    return self.state2input().shape
  
  @property
  def minScoreDelta(self):
    h, w = self.fog.shape
    return 1.0 / (h * w)