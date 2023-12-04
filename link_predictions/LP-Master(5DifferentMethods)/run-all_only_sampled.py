import networkx as nx
import matplotlib.pyplot as plt
from contextlib import closing
import pandas as pd
import link_prediction_scores as lp
from link_prediction_scores import calculate_all_scores_revised
from multiprocessing import Pool
import pickle, json
import os
import tensorflow as tf
import numpy as np

from scipy import sparse
path_update = r"/home2/xhe/updated_edges_revised//"
path_to_result = r"/home2/xhe/link-prediction-master/result_orig//"
sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'ForestFireSampler',
 'NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']


def auc_lpmaster(count):
    
    try:
        for i in range(1,6):

            net_name = "net"+str(count)+"__"
            A_ho = np.load(path_update + net_name+str(i)+"_Aho.npy")
            A_tr = np.load(path_update + net_name+str(i)+"_Atr.npy")
            A_orig = np.load(path_update + net_name+str(i)+"_Aorig.npy")
            A_test = np.load(path_update + net_name+str(i)+"_Atest.npy")
            A_val = np.load(path_update + net_name+str(i)+"_Aval.npy")


            # false validation edges 
            A_ho_aux = -1*A_ho + 1
            ng1_re = nx.Graph(A_ho_aux)
            a_val_graph = nx.Graph(A_val)
            adj_train = sparse.csr_matrix(A_ho)


            Nsamples = 10000 # number of samples
            edge_t = [] # list of true edges (positive samples)
            edge_f = [] # list of false edges (negative samples)
            for ll in range(Nsamples):

                #print(len(list(ng1.edges)))
                edge_t_idx_aux = np.random.randint(len(list(a_val_graph.edges)))
                edge_f_idx_aux = np.random.randint(len(list(ng1_re.edges)))
                edge_t.append([list(a_val_graph.edges)[edge_t_idx_aux][0],list(a_val_graph.edges)[edge_t_idx_aux][1]])
                edge_f.append([list(ng1_re.edges)[edge_f_idx_aux][0],list(ng1_re.edges)[edge_f_idx_aux][1]])



            for samp in sampling_methods:

                train_edges = np.loadtxt(path_update + net_name+str(i)+ samp +"_t_train.npy").astype('int')
                train_edges_false = np.loadtxt(path_update + net_name+str(i)+samp +"_f_train.npy").astype('int')
                test_edges = np.loadtxt(path_update + net_name+str(i)+samp +"_t_test.npy").astype('int')
                test_edges_false = np.loadtxt(path_update + net_name+str(i)+samp +"_f_test.npy").astype('int')

                val_edges, val_edges_false = edge_t, edge_f
                adj_sparse  = sparse.csr_matrix(A_orig)

                train_test_split = adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false
                testresult = calculate_all_scores_revised(adj_sparse, train_test_split, random_state=0, tf_dtype=tf.float32)

                with open(path_to_result+net_name+str(i)+ samp+ '_result.json', 'w') as fp:
                    json.dump(testresult, fp)
    except:
        pass

finished = np.loadtxt("finished_net.txt").astype("int")
netlist = finished[64:128]

#auc_lpmaster(netlist[0])

with closing(Pool(processes=len(netlist))) as pool:
    pool.map(auc_lpmaster, netlist)
    pool.terminate()

#with Pool(len(netlist)) as p:
    #print(p.map(auc_lpmaster, netlist))