#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from Agent.DQNEnsembleAgent import DQNEnsembleAgent
# limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1 * 1024)]
)

from Core.CMazeEnvironment import CMazeEnvironment, MazeActions
import numpy as np
import pygame
import pygame.locals as G
import random
from Agent.DQNAgent import DQNAgent
import glob
from collections import namedtuple
from model import createModel

def createMaze():
  sz = 16 * 4
  maze = (0.8 < np.random.rand(sz, sz)).astype(np.float32)
  res = CMazeEnvironment(
    maze=maze,
    pos=(0, 0),
    FOV=3,
    minimapSize=8
  )
  res.respawn()
  return res
  
class Colors:
  BLACK = (0, 0, 0)
  SILVER = (192, 192, 192)
  WHITE = (255, 255, 255)
  BLUE = (0, 0, 255)
  GREEN = (0, 255, 0)
  RED = (255, 0, 0)
  PURPLE = (255, 0, 255)

RLAgent = namedtuple('RLAgent', 'name agent environment')

class App:
  MODES = ['manual', 'random', 'agent']
  
  def __init__(self):
    self._running = True
    self._display_surf = None
    self._mode = 'manual'
    self._paused = True
    self._speed = 1
    self._agents = []
    self._activeAgent = 0
    self._createMaze()
    return
  
  def _createMaze(self):
    self._maze = createMaze()
    self._initMaze = self._maze.copy()
    if 'agent' == self._mode:
      self._assignMaze2Agents()
    return
  
  def on_init(self):
    pygame.init()
    
    self._display_surf = pygame.display.set_mode((850, 650), pygame.HWSURFACE)
    pygame.display.set_caption('Deep maze')
    self._font = pygame.font.Font(pygame.font.get_default_font(), 16)
    self._running = True
  
  def _assignMaze2Agents(self):
    agents = []
    for agent in self._agents:
      agents.append(RLAgent(
        agent.name, agent.agent,
        self._initMaze.copy()
      ))

    self._agents = agents
    return
  
  def _createNewAgent(self):
    self._agents = []
    models = []
    for i, x in enumerate(glob.iglob('weights/*.h5')):
      filename = os.path.abspath(x)
      model = createModel(shape=self._maze.input_size)
      model.load_weights(filename)
      models.append(model)
      agent = DQNAgent(model)
      name = os.path.basename(filename)
 
      self._agents.append(RLAgent(
        name[:-3], agent, self._initMaze.copy()
      ))
     
    self._agents.insert(0, RLAgent(
      'ensemble', 
      DQNEnsembleAgent(models),
      self._initMaze.copy()
    ))
    
    self._activeAgent = 0
    self._paused = True
    return

  def on_event(self, event):
    if event.type == G.QUIT:
      self._running = False

    if event.type == G.KEYDOWN:
      if G.K_ESCAPE == event.key:
        self._running = False
        
      if G.K_r == event.key:
        self._createMaze()
      # Switch mode
      if G.K_m == event.key:
        mode = next((i for i, x in enumerate(self.MODES) if x == self._mode))
        self._mode = self.MODES[(mode + 1) % len(self.MODES)]
        self._paused = True
        self._agents = []
        
        if 'agent' == self._mode:
          self._createNewAgent()
      #####
      if G.K_SPACE == event.key:
        self._paused = not self._paused
      #####
      if 'agent' == self._mode:
        if G.K_n == event.key:
          self._createNewAgent()

        if G.K_a == event.key:
          self._activeAgent = (self._activeAgent + 1) % len(self._agents)

      if 'manual' == self._mode:
        self._manualEvent(event)
      
      if not ('manual' == self._mode):
        if G.K_KP_PLUS == event.key:
          self._speed = min((32, 2 * self._speed))
        if G.K_KP_MINUS == event.key:
          self._speed = max((1, self._speed // 2))
          
    return

  def _manualEvent(self, event):
    actMapping = {
      G.K_LEFT: MazeActions.LEFT,
      G.K_RIGHT: MazeActions.RIGHT,
      G.K_UP: MazeActions.UP,
      G.K_DOWN: MazeActions.DOWN
    }
    
    act = actMapping.get(event.key, False)
    if act and self._maze.isPossible(act):
      self._maze.apply(act)
    return
   
  def on_loop(self):
    if self._paused: return
    
    if 'random' == self._mode:
      for _ in range(self._speed):
        actions = self._maze.validActions()
        if actions:
          self._maze.apply(random.choice(actions))
          
    if 'agent' == self._mode:
      for _ in range(self._speed):
        for agent in self._agents:
          maze = agent.environment 
          pred = agent.agent.process(maze.state2input(), maze.actionsMask())
          act = list(MazeActions)[pred]
          if maze.isPossible(act):
            maze.apply(act)
    pass
  
  def _renderMaze(self, env):
    fog = env.fog
    moves = env.moves
    maze = env.maze
    
    h, w = maze.shape
    dx, dy = delta = np.array([640, 640]) / np.array([w, h])
    for ix in range(w):
      for iy in range(h):
        isDiscovered = 0 < fog[ix, iy]
        isWall = 0 < maze[ix, iy]
        isWasHere = 0 < moves[ix, iy]
        y, x = delta * np.array([ix, iy])
        
        clr = Colors.WHITE
        if isWasHere: clr = Colors.GREEN
        if isWall: clr = Colors.PURPLE
        
        if not isDiscovered:
          clr = np.array(clr) * .3
        pygame.draw.rect(self._display_surf, clr, [x, y, dx - 1, dy - 1], 0)
    # current pos
    x, y = delta * env.pos
    pygame.draw.rect(self._display_surf, Colors.RED, [x, y, dx - 1, dy - 1], 0)
    return
  
  def _renderAgentsMaze(self):
    self._renderMaze(self._agents[self._activeAgent].environment)
    return
  
  def _drawText(self, text, pos, color):
    self._display_surf.blit(
      self._font.render(text, False, color),
      pos
    )
    return
  
  def _renderInfo(self):
    line = lambda i: (655, 15 + i * 20)
    
    self._drawText('Mode: %s' % (self._mode), line(0), Colors.BLUE) 
    if not ('agent' == self._mode):
      self._drawText(
        'Score: %.1f (%d)' % (self._maze.score * 100.0, self._maze.steps),
        line(1), Colors.BLUE
      ) 
      
    if 'random' == self._mode:
      self._drawText('Speed: x%.0f' % (self._speed), line(2), Colors.BLUE) 

    if 'agent' == self._mode:
      self._drawText('Speed: x%.0f' % (self._speed), line(1), Colors.BLUE)
      for i, agent in enumerate(self._agents):
        self._drawText(
          '%s%s | %.1f (%d)' % (
            '>> ' if i == self._activeAgent else '',
            agent.name, agent.environment.score * 100.0, agent.environment.steps
          ),
          line(2 + i), Colors.BLUE
        )
    return
  
  def on_render(self):
    self._display_surf.fill(Colors.SILVER)
    if 'agent' == self._mode:
      self._renderAgentsMaze()
    else:
      self._renderMaze(self._maze)
      
    self._renderInfo()
    pygame.display.flip()
 
  def run(self):
    if self.on_init() == False:
      self._running = False
      
    while self._running:
      for event in pygame.event.get():
        self.on_event(event)

      self.on_loop()
      self.on_render()
      
    pygame.quit()

def main():
  app = App()
  app.run()
  pass

if __name__ == '__main__':
  main()
