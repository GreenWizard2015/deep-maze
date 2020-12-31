import pylab as plt
import numpy as np
import math

def emulateBatch(testEnvs, agent, maxSteps):
  replays = [[] for _ in testEnvs]
  steps = 0
  while (steps < maxSteps) and not all(e.done for e in testEnvs):
    steps += 1
    
    activeEnvs = [(i, e) for i, e in enumerate(testEnvs) if not e.done]

    states = [e.state for _, e in activeEnvs]
    actionsMasks = [e.actionsMask() for _, e in activeEnvs] 
    actions = agent.processBatch(states, actionsMasks)
    
    for (i, e), action, actionsMask in zip(activeEnvs, actions, actionsMasks):
      state, reward, done, prevState = e.apply(action)
      replays[i].append((prevState, action, reward, actionsMask))
      if done: # save last state with dummy data
        replays[i].append((state, action, 0, actionsMask))

  return [(replay, e.done) for replay, e in zip(replays, testEnvs)]

def normalizeRewards(replay):
  prevStates, actions, rewards, actionsMasks = zip(*replay)
  rewards = np.array(rewards)
  
  std = rewards.std()
  std = 1 if 0 == std else std
  rewards = (rewards - rewards.mean()) / std
  return list(zip(prevStates, actions, rewards, actionsMasks))

def clipReplay(replay, loopLimit):
  if loopLimit < len(replay):
    # find and cutoff loops
    tail = replay[-loopLimit:]
    lastState = replay[-1][0]
    ind = next((i for i, step in enumerate(tail) if np.array_equal(step[0], lastState)), len(tail)) - len(tail)
    return replay[ind:]
  return replay
  
def trackScores(scores, metrics, levels=[.1, .3, .5, .75, .9], metricName='scores'):
  if metricName not in metrics:
    metrics[metricName] = {}
    
  def series(name):
    if name not in metrics[metricName]:
      metrics[metricName][name] = []
    return metrics[metricName][name]
  ########
  N = len(scores)
  orderedScores = list(sorted(scores, reverse=True))
  totalScores = sum(scores) / N
  series('avg.').append(totalScores)
  
  for level in levels:
    series('top %.0f%%' % (level * 100)).append(orderedScores[int(N * level)])
  return

def plotData2file(data, filename, maxCols=3):
  plt.clf()
  N = len(data)
  rows = (N + maxCols - 1) // maxCols
  cols = min((N, maxCols))
  
  figSize = plt.rcParams['figure.figsize']
  fig = plt.figure(figsize=(figSize[0] * cols, figSize[1] * rows))
  
  axes = fig.subplots(ncols=cols, nrows=rows)
  axes = axes.reshape((-1,)) if 1 < len(data) else [axes]
  for (chartname, series), axe in zip(data.items(), axes):
    for name, dataset in series.items():
      axe.plot(dataset, label=name)
    axe.title.set_text(chartname)
    axe.legend()
    
  fig.savefig(filename)
  plt.close(fig)
  return