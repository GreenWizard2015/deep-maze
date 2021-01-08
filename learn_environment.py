from Core.MazeRLWrapper import MazeRLWrapper
from Utils.ExperienceBuffers.CebPrioritized import CebPrioritized
from Agent.DQNAgent import DQNAgent
import time
import Utils
import fit_stage
import os

def learn_environment(model, params):
  NAME = params['name']
  BATCH_SIZE = params['batch size']
  GAMMA = params['gamma']
  BOOTSTRAPPED_STEPS = params['bootstrapped steps']
  metrics = {}

  environments = [
    MazeRLWrapper(params['maze']) for _ in range(params['test episodes'])
  ]
  
  memory = CebPrioritized(maxSize=5000, sampleWeight='abs')
  ######################################################
  def testModel(EXPLORE_RATE):
    for e in environments: e.reset()
    replays = [replay for replay, _ in Utils.emulateBatch(
        environments,
        DQNAgent(model, exploreRate=EXPLORE_RATE, noise=params.get('agent noise', 0)),
        maxSteps=params.get('max test steps')
      )
    ]
    
    ################
    # explore if hit the loop
    envsIndexes = [i for i, e in enumerate(environments) if e.hitTheLoop]
    if envsIndexes:
      envs = [environments[i] for i in envsIndexes]
      for e in envs: e.Continue()
      exploreReplays = Utils.emulateBatch(
        envs,
        DQNAgent(
          model,
          exploreRate=params.get('explore rate after loop', 1),
          noise=params.get('agent noise after loop', 0)
        ),
        maxSteps=params.get('max steps after loop', 16)
      )
      for ind, (replay, _) in zip(envsIndexes, exploreReplays):
        replays[ind] += replay[1:]
    ################
    for replay in replays:
      if BOOTSTRAPPED_STEPS < len(replay):
        memory.addEpisode(replay, terminated=True)
    return [x.score for x in environments]
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