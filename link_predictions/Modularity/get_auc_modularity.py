# +
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
import json
import warnings
warnings.filterwarnings('ignore')
path = r"/home/xhe/after_samp_process/link_predictions/Modularity//"



sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'ForestFireSampler',
 'NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

# +
auc_aa = []
for net in range(572):
    
    temp_aa = []
        
    for count in range(1,6):
        temp = np.loadtxt(path + "m" + str(count) + "/net" + str(net) + "__modularity_auc.txt", delimiter=',',)
        temp_aa.append(temp)
            
    auc_aa.append(temp_aa)
    

np.save("auc_modularity.npy", auc_aa)
