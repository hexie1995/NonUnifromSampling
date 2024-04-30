import networkx as nx
import matplotlib.pyplot as plt
from contextlib import closing
import pandas as pd
import link_prediction_scores as lp
from link_prediction_scores import calculate_all_scores_revised
from multiprocessing import Pool
import pickle, json
import os
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

# +
from scipy import sparse
path_update = r"/home/xhe/sampled_edges_2024_old//"
path_to_result = r"/home/xhe/after_samp_process/link_predictions/LP-Master/results//"
sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler', 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'ForestFireSampler', 'NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler','RandomWalkSampler', 
 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler','RandomNodeNeighborSampler', 'ShortestPathSampler']

    
do_not_touch = ['net10_', 'net128_','net12_', 'net13_','net147_', 'net18_','net22_','net282_','net298_','net307_','net308_',
                'net310_','net312_','net318_', 'net323_','net325_','net329_','net331_','net335_', 'net33_', 'net341_', 'net345_',
                'net347_','net348_','net349_', 'net34_','net357_','net361_', 'net362_','net364_','net36_','net37_','net38_','net397_',
                'net39_','net400_','net403_','net412_', 'net416_','net417_','net421_','net432_','net435_','net436_','net437_',
                'net440_','net443_','net453_','net45_','net513_','net515_','net568_','net570_','net571_','net79_','net8_', 'net9_']


# -

def auc_lpmaster(count):
    net_name = "net"+str(count)+"_"
    #num_nodes = DATA["number_nodes"][net]

    if net_name not in do_not_touch:
        try:
            for i in range(1,6):

                A_ho = np.load(path_update + net_name+"_"+str(i)+"_Aho.npy")
                A_orig = np.load(path_update + net_name+"_"+str(i)+"_Aorig.npy")
                A_test = np.load(path_update + net_name+"_"+str(i)+"_Atest.npy")

                train_edges_false = np.loadtxt(path_update + net_name+"_"+str(i) +"_f_train_10000.npy").astype('int')
                test_edges = np.loadtxt(path_update + net_name+"_"+str(i) +"_t_test_10000.npy").astype('int')
                test_edges_false = np.loadtxt(path_update + net_name+"_"+str(i) +"_f_test_10000.npy").astype('int')
                val_edges_false = np.loadtxt(path_update + net_name+"_" + str(i) + "_f_valid_10000.npy").astype('int')

                for samp in sampling_methods:
                    A_val = np.load(path_update + net_name+"_"+str(i) + samp + "_Aval.npy")
                    A_tr = np.load(path_update + net_name+"_"+str(i) + samp + "_Atr.npy")

                    adj_train = sparse.csr_matrix(A_tr)

                    train_edges = np.loadtxt(path_update + net_name+"_"+str(i)+ samp +"_t_train.npy").astype('int')
                    val_edges = np.loadtxt(path_update + net_name+"_"+str(i)+ samp + "_t_valid.npy").astype('int') 

                    adj_sparse  = sparse.csr_matrix(A_orig)

                    train_test_split = adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false
                    testresult = calculate_all_scores_revised(adj_sparse, train_test_split, random_state= i )

                    with open(path_to_result+net_name+"_"+str(i)+ samp+ '_result.json', 'w') as fp:
                        json.dump(testresult, fp)
        except:
            with open('not_processed.txt', 'a') as fd:
                fd.write('\n')
                fd.write(str(net_name))


finished = np.loadtxt("finished_net.txt").astype("int")
#netlist = finished[64:128]
netlist = list(range(0,100))

# +
#auc_lpmaster(netlist[1])
# -

with closing(Pool(processes=len(netlist))) as pool:
    pool.map(auc_lpmaster, netlist)
    pool.terminate()

#with Pool(len(netlist)) as p:
    #print(p.map(auc_lpmaster, netlist))
