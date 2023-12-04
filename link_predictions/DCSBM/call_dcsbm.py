from MDL_DCSBM_code_function import *
#sys.path.append('/Users/amgh5286/stacking/new_real_data/ST_algs/LD/Infomap/Infomap/examples/python/')
import time
import pickle
import numpy as np
from multiprocessing import Pool

path = r"/home/xhe/updated_edges_revised//"
savepath = r"/home/xhe/DCSBM_5_RUN/dcsbm_1//"

sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'DiffusionTreeSampler', 'ForestFireSampler',
 'CommonNeighborAwareRandomWalkSampler','NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 
 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler','CommunityStructureExpansionSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

def auc_dcsbm(count):
    net = count
    auc_for_net = []
    name = "net"+ str(net) +"_"
    for samp in sampling_methods:
        try:
            auc = mdldcsbm_for_sampling(path, net, samp)
            auc_for_net.append(auc)
        except:
            auc_for_net.append(0)
    np.save(savepath + name + "_dcsbm_auc.npy", auc_for_net)

netlist = list(range(512,572))

for i in range(572):
    auc_dcsbm(i)

#with Pool(len(netlist)) as p:
#    print(p.map(auc_dcsbm, netlist))
