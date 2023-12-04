from OLP import topol_stacking_for_sampling
import pickle
import numpy as np
from multiprocessing import Pool
path = r"/home/xhe/updated_edges_revised//"
savepath = r"/home/xhe/OLP_5_RUN/olp_1//"

sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'DiffusionTreeSampler', 'ForestFireSampler',
 'CommonNeighborAwareRandomWalkSampler','NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 
 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler','CommunityStructureExpansionSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

def auc_olp(count):
    net = count
    auc_for_net = []
    pre_for_net = []
    rec_for_net = []

    name = "net"+ str(net) +"_"

        
    for samp in sampling_methods:
        try:
            print(samp)
            auc, precision, recall = topol_stacking_for_sampling(path, net, samp)
            auc_for_net.append(auc)
            pre_for_net.append(precision)
            rec_for_net.append(recall)
        except:
            auc_for_net.append(0)
            pre_for_net.append([0,0,0])
            rec_for_net.append([0,0,0])

    print(auc_for_net)
    np.save(savepath + name + "_olp_auc.npy", auc_for_net)
    np.save(savepath + name + "_olp_precision.npy", pre_for_net)
    np.save(savepath + name + "_olp_recall.npy", rec_for_net)

netlist = list(range(400,572))

with Pool(len(netlist)) as p:
    print(p.map(auc_olp, netlist))