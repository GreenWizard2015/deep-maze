from enum import Enum
import numpy as np

class MazeActions(Enum):
  LEFT = (-1, 0)
  RIGHT = (1, 0)
  UP = (0, -1)
  DOWN = (0, 1)

class CMazeEnviroment:
  def __init__(self, maze, pos, FOV):
    self._maze = np.pad(np.array(maze), FOV, constant_values=(1,))
    self._pos = np.array(pos) + FOV
    self._fov = FOV
    
    self._fog = np.zeros_like(self._maze)
    self._updateFog()
  
  def _updateFog(self):
    y, x = self._pos
    self._fog[
      x - self._fov:x + self._fov + 1,
      y - self._fov:y + self._fov + 1
    ] = 1
    return
  
  def apply(self, action):
    self._pos += action.value
    self._updateFog()
    return

  def vision(self):
    y, x = self._pos
    return self._maze[
      x - self._fov:x + self._fov + 1,
      y - self._fov:y + self._fov + 1
    ]
  
  @property
  def state(self):
    return ((self.vision(), self._fog, ), self.score, self.done)
  
  @property
  def done(self):
    y, x = self._pos
    return 1 < self._maze[x, y]
  
  @property
  def score(self):
    h, w = self._fog.shape
    total = h * w
    return np.count_nonzero(self._fog) / total