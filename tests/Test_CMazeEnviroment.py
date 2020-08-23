from verify import expect
from Core.CMazeEnviroment import CMazeEnviroment, MazeActions
import numpy as np

class Test_CMazeEnviroment:
  def test_vision(self):
    env = CMazeEnviroment(
      maze=[
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 2, 0],
      ],
      pos=(0, 0),
      FOV=1
    )
    
    valid = np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ])
    expect(str(env.vision())).is_equal(str(valid))
    
  def test_apply(self):
    env = CMazeEnviroment(
      maze=[
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 2, 0],
      ],
      pos=(0, 0),
      FOV=1
    )
    env.apply(MazeActions.RIGHT)
    env.apply(MazeActions.DOWN)
    env.apply(MazeActions.RIGHT)
    env.apply(MazeActions.DOWN)
    env.apply(MazeActions.RIGHT)
    
    valid = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
    ])
    expect(str(env.vision())).is_equal(str(valid))
    
  def test_increasingScoreWhileExploring(self):
    env = CMazeEnviroment(
      maze=[
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 2, 0],
      ],
      pos=(0, 0),
      FOV=1
    )
    
    oldScore = env.score
    env.apply(MazeActions.RIGHT)
    newScore = env.score
    expect(oldScore).is_less(newScore)
    
  def test_scoreNotChanged(self):
    env = CMazeEnviroment(
      maze=[
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 2, 0],
      ],
      pos=(0, 0),
      FOV=1
    )
    
    env.apply(MazeActions.RIGHT)
    oldScore = env.score
    env.apply(MazeActions.LEFT)
    newScore = env.score
    expect(oldScore).is_equal(newScore)
    