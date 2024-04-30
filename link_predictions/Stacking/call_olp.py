from OLP import topol_stacking_for_sampling
import pickle
import numpy as np
from multiprocessing import Pool
path1 = r"/home/xhe/updated_edges_revised//"
path =  r"/home/xhe/sampled_edges_2024_old//"
savepath = r"/home/xhe/after_samp_process/link_predictions/Stacking/m1//"

sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithInduction', 'DiffusionSampler', 
 'ForestFireSampler','NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler','RandomWalkSampler', 
 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 'CirculatedNeighborsRandomWalkSampler', 
 'BreadthFirstSearchSampler','DepthFirstSearchSampler', 'RandomWalkWithJumpSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

def auc_olp(count):
    net = count
    auc_for_net = []
    pre_for_net = []
    rec_for_net = []

    name = "net"+ str(net) +"_"
      
    for samp in sampling_methods:
  
        print(samp)
        auc, precision, recall = topol_stacking_for_sampling(path, path1, net, samp)
        auc_for_net.append(auc)
        pre_for_net.append(precision)
        rec_for_net.append(recall)

    print(auc_for_net)
    np.save(savepath + name + "_olp_auc.npy", auc_for_net)
    np.save(savepath + name + "_olp_precision.npy", pre_for_net)
    np.save(savepath + name + "_olp_recall.npy", rec_for_net)

netlist = np.loadtxt("finished_net.txt").astype(int).tolist()
#netlist = netlist

with Pool(len(netlist)) as p:
    print(p.map(auc_olp, netlist))
