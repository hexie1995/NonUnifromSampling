# +
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
import json
import warnings
warnings.filterwarnings('ignore')
path = r"/home/xhe/after_samp_process/link_predictions/Stacking//"



sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'DiffusionTreeSampler', 'ForestFireSampler',
 'CommonNeighborAwareRandomWalkSampler','NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 
 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler','CommunityStructureExpansionSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

# +
auc_aa = []
for net in range(572):
    
    temp_aa = []
    
    for count in range(1,5):
        try:
                    temp = np.load(path + "m" + str(count) + "/net" + str(net) + "__olp_auc.npy")
        except:
            temp = [np.nan]*20
        temp_aa.append(temp)
            
    auc_aa.append(temp_aa)
    

np.save("auc_olp.npy", auc_aa)
