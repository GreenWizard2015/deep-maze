from enum import Enum
import numpy as np
import random

class MazeActions(Enum):
  LEFT = (-1, 0)
  RIGHT = (1, 0)
  UP = (0, -1)
  DOWN = (0, 1)

class CMazeEnviroment:
  def __init__(self, maze, pos, FOV):
    self.maze = np.pad(np.array(maze), FOV, constant_values=(1,))
    self._fov = FOV
    
    x, y = np.array(pos) + FOV
    self.spawnAt(x, y)
    return

  def spawnAt(self, x, y):
    self.pos = np.array([y, x])
    self.fog = np.zeros_like(self.maze)
    self._updateFog()
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
  
  def _updateFog(self):
    y, x = self.pos
    self.fog[
      x - self._fov:x + self._fov + 1,
      y - self._fov:y + self._fov + 1
    ] = 1
    return
  
  def apply(self, action):
    self.pos += action.value
    self._updateFog()
    return

  def vision(self):
    y, x = self.pos
    return self.maze[
      x - self._fov:x + self._fov + 1,
      y - self._fov:y + self._fov + 1
    ]
  
  @property
  def state(self):
    return ((self.vision(), self.fog, ), self.score, self.done)
  
  @property
  def done(self):
    y, x = self._pos
    return 1 < self.maze[x, y]
  
  @property
  def score(self):
    h, w = self.fog.shape
    total = h * w
    return np.count_nonzero(self.fog) / total
  
  def copy(self):
    # dirty copy
    res = CMazeEnviroment(self.maze, self.pos, self._fov)
    res.maze = self.maze.copy()
    res.fog = self.fog.copy()
    res.pos = self.pos.copy()
    return res
  
  def isPossible(self, action):
    y, x = self.pos + action.value
    return self.maze[x, y] <= 0
  
  def validActions(self):
    return [ act for act in MazeActions if self.isPossible(act) ]