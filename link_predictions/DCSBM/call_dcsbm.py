from MDL_DCSBM_code_function import *
#sys.path.append('/Users/amgh5286/stacking/new_real_data/ST_algs/LD/Infomap/Infomap/examples/python/')
import time
import pickle
import numpy as np
from multiprocessing import Pool

path1 = r"/home/xhe/updated_edges_revised//"
path =  r"/home/xhe/sampled_edges_2024_old//"
savepath = r"/home/xhe/after_samp_process/link_predictions/DCSBM/m1//"

sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithInduction', 'DiffusionSampler', 
 'ForestFireSampler','NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler','RandomWalkSampler', 
 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 'CirculatedNeighborsRandomWalkSampler', 
 'BreadthFirstSearchSampler','DepthFirstSearchSampler', 'RandomWalkWithJumpSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

def auc_dcsbm(count):
    net = count
    auc_for_net = []
    name = "net"+ str(net) +"_"
    for samp in sampling_methods:
        #try:
        auc = mdldcsbm_for_sampling(path, path1, net, samp)
        auc_for_net.append(auc)
        #except:
        #    auc_for_net.append(0)
    np.save(savepath + name + "_dcsbm_auc.npy", auc_for_net)

#for i in range(1):
#    auc_dcsbm(i)

netlist = np.loadtxt("finished_net.txt").astype(int).tolist()
#netlist = netlist

with Pool(len(netlist)) as p:
    print(p.map(auc_dcsbm, netlist))
