from Core.MazeRLWrapper import MazeRLWrapper
from Utils.ExperienceBuffers.CebPrioritized import CebPrioritized
from Agent.DQNAgent import DQNAgent
import time
import Utils
import fit_stage
import os
from Utils.ExperienceBuffers.CebLinear import CebLinear

def learn_environment(model, params):
  NAME = params['name']
  BATCH_SIZE = params['batch size']
  GAMMA = params['gamma']
  BOOTSTRAPPED_STEPS = params['bootstrapped steps']
  LOOP_LIMIT = params['maze']['loop limit']
  metrics = {}

  environments = [
    MazeRLWrapper(params['maze']) for _ in range(params['test episodes'])
  ]
  
  memory = CebPrioritized(maxSize=5000, sampleWeight='abs')
  doomMemory = CebLinear(
    maxSize=params.get('max steps after loop', 16) * 1000,
    sampleWeight='abs'
  )
  
  ######################################################
  def testModel(EXPLORE_RATE):
    for e in environments: e.reset()
    replays = Utils.emulateBatch(
      environments,
      DQNAgent(model, exploreRate=EXPLORE_RATE, noise=params.get('agent noise', 0)),
      maxSteps=params.get('max test steps')
    )
    for replay, _ in replays:
      if params.get('clip replay', False):
        replay = Utils.clipReplay(replay, loopLimit=LOOP_LIMIT)
      if BOOTSTRAPPED_STEPS < len(replay):
        memory.addEpisode(replay, terminated=True)
    
    scores = [x.score for x in environments]
    ################
    # collect bad experience
    envs = [e for e in environments if e.hitTheLoop]
    if envs:
      for e in envs: e.Continue()
      replays = Utils.emulateBatch(
        envs,
        DQNAgent(
          model,
          exploreRate=params.get('explore rate after loop', 1),
          noise=params.get('agent noise after loop', 0)
        ),
        maxSteps=params.get('max steps after loop', 16)
      )
      for replay, _ in replays:
        if BOOTSTRAPPED_STEPS < len(replay):
          doomMemory.addEpisode(replay, terminated=True)
    ################
    return scores
  ######################################################
  # collect some experience
  for _ in range(2):
    testModel(EXPLORE_RATE=0)
  #######################
  bestModelScore = -float('inf')
  for epoch in range(params['epochs']):
    T = time.time()
    
    EXPLORE_RATE = params['explore rate'](epoch)
    alpha = params.get('alpha', lambda _: 1)(epoch)
    print(
      '[%s] %d/%d epoch. Explore rate: %.3f. Alpha: %.5f.' % (NAME, epoch, params['epochs'], EXPLORE_RATE, alpha)
    )
    ##################
    # Training
    trainLoss = fit_stage.train(
      model, memory,
      {
        'gamma': GAMMA,
        'batchSize': BATCH_SIZE,
        'steps': BOOTSTRAPPED_STEPS,
        'episodes': params['train episodes'](epoch),
        'alpha': alpha
      }
    )
    print('Avg. train loss: %.4f' % trainLoss)
    
    trainLoss = fit_stage.train(
      model, doomMemory,
      {
        'gamma': GAMMA,
        'batchSize': BATCH_SIZE,
        'steps': BOOTSTRAPPED_STEPS,
        'episodes': params['train doom episodes'](epoch),
        'alpha': params.get('doom alpha', lambda _: alpha)(epoch)
      }
    )
    print('Avg. train doom loss: %.4f' % trainLoss)
    ##################
    # test
    print('Testing...')
    scores = testModel(EXPLORE_RATE)
    Utils.trackScores(scores, metrics)
    ##################
    
    scoreSum = sum(scores)
    print('Scores sum: %.5f' % scoreSum)
    if (bestModelScore < scoreSum) and (params['warm up epochs'] < epoch):
      print('save best model (%.2f => %.2f)' % (bestModelScore, scoreSum))
      bestModelScore = scoreSum
      model.save_weights('weights/%s.h5' % NAME)
    ##################
    os.makedirs('charts', exist_ok=True)
    Utils.plotData2file(metrics, 'charts/%s.jpg' % NAME)
    print('Epoch %d finished in %.1f sec.' % (epoch, time.time() - T))
    print('------------------')