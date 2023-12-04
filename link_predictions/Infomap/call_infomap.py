# +
from Infomap_code_function import *
import infomap
import time
import pickle
import numpy as np
from multiprocessing import Pool
path = r"/home/xhe/updated_edges_revised//"
savepath = r"/home/xhe/INFOMAP_5_RUN/infomap_1//"

sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'DiffusionTreeSampler', 'ForestFireSampler',
 'CommonNeighborAwareRandomWalkSampler','NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 
 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler','CommunityStructureExpansionSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

def auc_infomap(count):
    net = count
    infomap_auc = []
    name = "net"+ str(net) +"_"
    for samp in sampling_methods:
        try:
            auc = infomap_for_sampling(path, net, samp)
        except:
            auc = 0
            print("did not pass sampling")
        infomap_auc.append(auc)
    np.save(savepath + name + "_infomap_auc.npy", infomap_auc)

#netlist = list(range(572))

for net in range(572):
    auc_infomap(net)


#with Pool(len(netlist)) as p:
#    print(p.map(auc_infomap, netlist))# +
