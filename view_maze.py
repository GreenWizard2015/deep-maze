#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Core.CMazeEnviroment import CMazeEnviroment, MazeActions
import numpy as np
import pygame
import pygame.locals as G
import random

def createMaze():
  sz = 64
  maze = (0.8 < np.random.rand(sz, sz)).astype(np.float32)
  res = CMazeEnviroment(
    maze=maze,
    pos=(0, 0),
    FOV=3
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

class App:
  MODES = ['manual', 'random']
  def __init__(self):
    self._running = True
    self._display_surf = None
    self._createMaze()
    self._mode = 'manual'
    self._paused = True
    self._speed = 20
  
  def _createMaze(self):
    self._maze = createMaze()
    self._initMaze = self._maze.copy()
    return
  
  def on_init(self):
    pygame.init()
    
    self._display_surf = pygame.display.set_mode((800, 650), pygame.HWSURFACE)
    pygame.display.set_caption('Deep maze')
    self._font = pygame.font.Font(pygame.font.get_default_font(), 16)
    self._running = True
 
  def on_event(self, event):
    if event.type == G.QUIT:
      self._running = False

    if event.type == G.KEYDOWN:
      if G.K_m == event.key:
        mode = next((i for i, x in enumerate(self.MODES) if x == self._mode))
        self._mode = self.MODES[(mode + 1) % len(self.MODES)]
        self._paused = True
        
      if G.K_SPACE == event.key:
        self._paused = not self._paused

      if G.K_ESCAPE == event.key:
        self._running = False
      
      if 'manual' == self._mode:  
        if G.K_r == event.key:
          self._createMaze()
          
        if G.K_i == event.key:
          self._maze = self._initMaze.copy()
          
        if G.K_y == event.key:
          self._maze.respawn()
          
        actMapping = {
          G.K_LEFT: MazeActions.LEFT,
          G.K_RIGHT: MazeActions.RIGHT,
          G.K_UP: MazeActions.UP,
          G.K_DOWN: MazeActions.DOWN
        }
        
        act = actMapping.get(event.key, False)
        if act and self._maze.isPossible(act):
          self._maze.apply(act)
      #####
    return
 
  def on_loop(self):
    if ('random' == self._mode) and not self._paused:
      for _ in range(self._speed):
        actions = self._maze.validActions()
        if actions:
          self._maze.apply(random.choice(actions))
    pass
  
  def _renderMaze(self):
    fog = self._maze.fog
    maze = self._maze.maze
    h, w = maze.shape
    dx, dy = delta = np.array([640, 640]) / np.array([w, h])
    for ix in range(w):
      for iy in range(h):
        isDiscovered = 0 < fog[ix, iy]
        isWall = 0 < maze[ix, iy]
        y, x = delta * np.array([ix, iy])
        
        clr = Colors.PURPLE if isWall else Colors.WHITE
        if not isDiscovered:
          clr = np.array(clr) * .3
        pygame.draw.rect(self._display_surf, clr, [x, y, dx - 1, dy - 1], 0)
    # current pos
    x, y = delta * self._maze.pos
    pygame.draw.rect(self._display_surf, Colors.RED, [x, y, dx - 1, dy - 1], 0)
    return
  
  def _renderInfo(self):
    self._display_surf.blit(
      self._font.render(
        'Score: %.2f' % (self._maze.score),
        False, Colors.BLUE
      ), (655, 15)
    )
    
    self._display_surf.blit(
      self._font.render(
        'Mode: %s' % (self._mode),
        False, Colors.BLUE
      ), (655, 35)
    )
    return
  
  def on_render(self):
    self._display_surf.fill(Colors.SILVER)
    self._renderMaze()
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
