import numpy as np
import pandas as pd
import time
import networkx as nx
import pickle
import random
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
import littleballoffur as lbf
from littleballoffur import RandomNodeSampler, DegreeBasedSampler, PageRankBasedSampler,RandomEdgeSampler
from littleballoffur import RandomNodeEdgeSampler,HybridNodeEdgeSampler,RandomEdgeSamplerWithPartialInduction,RandomEdgeSamplerWithInduction
from littleballoffur import RandomEdgeSamplerWithInduction, DiffusionSampler,DiffusionTreeSampler, ForestFireSampler, SpikyBallSampler
from littleballoffur import CommonNeighborAwareRandomWalkSampler,NonBackTrackingRandomWalkSampler, LoopErasedRandomWalkSampler
from littleballoffur import RandomWalkSampler, RandomWalkWithRestartSampler,MetropolisHastingsRandomWalkSampler, SnowBallSampler
from littleballoffur import CirculatedNeighborsRandomWalkSampler, BreadthFirstSearchSampler,DepthFirstSearchSampler, RandomWalkWithJumpSampler
from littleballoffur import CommunityStructureExpansionSampler, FrontierSampler,RandomNodeNeighborSampler, ShortestPathSampler
with open(r"CommunityFitNet_updated.pickle", "rb") as input_file:
    DATA = pickle.load(input_file)
from multiprocessing import Pool 
savepath = r"/home2/xhe/updated_edges_revised//"
EDGELIST = DATA["edges_id"]
sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'DiffusionTreeSampler', 'ForestFireSampler', 'SpikyBallSampler',
 'CommonNeighborAwareRandomWalkSampler','NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 'SnowBallSampler',
 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler','CommunityStructureExpansionSampler', 'FrontierSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']
len(EDGELIST)

for samp in sampling_methods:
    for bi in ["t","f"]:
        for tt in ["train", "test"]:
            DATA[samp +"_"+ bi +"_" + tt] = [[]] * DATA.shape[0]
            DATA[samp +"_"+ bi +"_" + tt + "_10000"] = [[]] * DATA.shape[0]

DATA["edge_set_test"] = [[]] * DATA.shape[0]
DATA["edge_set_train"] = [[]] * DATA.shape[0]
for net in range(572):
    G = nx.Graph()
    G.add_edges_from(EDGELIST[net])
    
    x = EDGELIST[net]
    x = np.unique(x,axis=0)
    np.random.shuffle(x)
    #print(x)
    x1 = np.array_split(x, 5)
    count = 0
    x = x.tolist()
    
    y = x1[0]
    y = y.tolist()
    comp = [item for item in x if item not in y]
    
    test_name = "edge_set" + "_test"
    train_name = "edge_set" + "_train"

    DATA[test_name][net]= np.array(y)
    DATA[train_name][net] = np.array(comp)

sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'ForestFireSampler',
 'NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

do_not_touch = ['net10_', 'net128_','net12_', 'net13_','net147_', 'net18_','net22_','net282_','net298_','net307_','net308_',
                'net310_','net312_','net318_', 'net323_','net325_','net329_','net331_','net335_', 'net33_', 'net341_', 'net345_',
                'net347_','net348_','net349_', 'net34_','net357_','net361_', 'net362_','net364_','net36_','net37_','net38_','net397_',
                'net39_','net400_','net403_','net412_', 'net416_','net417_','net421_','net432_','net435_','net436_','net437_',
                'net440_','net443_','net453_','net45_','net513_','net515_','net568_','net570_','net571_','net79_','net8_', 'net9_']

not_processed = []

def adj_to_nodes_edges(A):
    
    """ 
    This function change adjacency matrix to list of nodes and edges.

    Input and Parameters:
    -------
    A: the adjacency matrix

    Returns:
    -------
    nodes: node list of the given network
    edges: edge list of the given network

    Examples:
    -------
    >>> nodes, edges = adj_to_nodes_edges(A)
    """
    
    num_nodes = A.shape[0]
    nodes = range(num_nodes)
    edges = np.where(np.triu(A,1))
    row = edges[0]
    col = edges[1]
    edges = np.vstack((row,col)).T
    return nodes, edges


def gen_tr_ho_networks(A_orig, alpha, alpha_, nsim_id):
    
    np.random.seed(nsim_id)
    A_ho = 1*(np.triu(A_orig,1)==1)
    rows_one, cols_one = np.where(np.triu(A_ho,1))
    ones_prob_samp = np.random.binomial(1, size=len(rows_one), p=alpha)
    A_ho[rows_one, cols_one] = ones_prob_samp
    A_ho = A_ho + A_ho.T
    

    A_tr = 1*(np.triu(A_ho,1)==1)
    rows_one, cols_one = np.where(np.triu(A_tr,1))
    ones_prob_samp = np.random.binomial(1, size=len(rows_one), p=alpha_)
    A_tr[rows_one, cols_one] = ones_prob_samp
    A_tr = A_tr + A_tr.T

    # A_hold out is the true training matrix, i.e the whole training set 
    # A_tr is THE used training matrix, because it is the edge set that we sampled from 
    # A_hold-A_tr is the validation set that we used to validate stuff
    # A_orig - A_ho is the test set
    A_test = A_orig - A_ho
    A_val = A_ho - A_tr
    return A_ho, A_tr, A_test, A_orig, A_val


def sample_true_false_edges(A_orig, A_tr, A_ho, nsim_id):  
    nodes, edge_tr = adj_to_nodes_edges(A_tr)
    #nsim_id = 0
    np.random.seed(nsim_id)

    A_diff = A_tr
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_ho_aux = -1*A_ho + 1
    ne_ho = sparse.find(sparse.triu(A_ho_aux,1)) # false candidates
    
    train_tr = []
    train_fl = []
    for ll in range(len(e_diff[0])):
        edge_t_idx_aux = ll
        train_tr.append((e_diff[0][ll],e_diff[1][ll]))
    
    
    for ll in range(len(ne_ho[0])):
        edge_t_idx_aux = ll
        train_fl.append((ne_ho[0][ll],ne_ho[1][ll]))
    
    
    
    A_diff = A_orig - A_ho
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_orig_aux = -1*A_orig + 1
    ne_orig = sparse.find(sparse.triu(A_orig_aux,1)) # false candidates
    
    
    test_tr = []
    test_fl = []
    for ll in range(len(e_diff[0])):
        edge_t_idx_aux = ll
        test_tr.append((e_diff[0][ll],e_diff[1][ll]))
    
    
    for ll in range(len(ne_orig[0])):
        edge_t_idx_aux = ll
        test_fl.append((ne_orig[0][ll],ne_orig[1][ll]))
    

    
    return train_tr, train_fl, test_tr, test_fl


def generate_graphs(true_edges, false_edges):
        G = nx.Graph()
        G.add_edges_from(true_edges)
        #print(G.edges)

        #print(list(nx.connected_components(G)))
        ccc = max(list(nx.connected_components(G)), key=len)
        #print(ccc)
        largeG = G.subgraph(ccc).copy()
        #print(largeG.edges)



        nn = len(largeG.nodes) - 1
        #nonedges = list(nx.non_edges(G))

        G_reverse = nx.Graph()
        G_reverse.add_edges_from(false_edges)

        #print(list(nx.connected_components(G_reverse)))
        ccc = max(list(nx.connected_components(G_reverse)), key=len)
        #print(ccc)
        largeG_reverse = G_reverse.subgraph(ccc).copy()
        nnr = len(largeG_reverse.nodes) - 1


        index_map = dict(enumerate(list(largeG.nodes)))
        inv_map = {v: k for k, v in index_map.items()}
        G1 = nx.relabel_nodes(largeG, inv_map)

        #print(nn)
        #print(nnr)

        index_map_re = dict(enumerate(list(largeG_reverse.nodes)))
        inv_map_re = {v: k for k, v in index_map_re.items()}
        G1_re = nx.relabel_nodes(largeG_reverse, inv_map_re)
        
        return nn, nnr, G1, G1_re, index_map, index_map_re


def process_net(net):
    
    G = nx.Graph()
    G.add_edges_from(EDGELIST[net])    
    name = "net"+ str(net) + "_"

    edges_orig = EDGELIST[net]
    np.savetxt(savepath + name +'.txt', edges_orig)
    edges_orig = np.array(np.matrix(edges_orig))
    
    num_nodes = int(np.max(edges_orig)) + 1
    row = np.array(edges_orig)[:,0]
    col = np.array(edges_orig)[:,1]

    data_aux = np.ones(len(row))
    A_orig = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A_orig = sparse.triu(A_orig,1) + sparse.triu(A_orig,1).transpose()
    A_orig[A_orig>0] = 1 
    A_orig = A_orig.todense()
    
    #print(A_orig)
    for repeated in range(1,6):
    
        A_ho, A_tr, A_test, A_orig, A_val = gen_tr_ho_networks(A_orig, 0.8, 0.8, repeated) 
        #A_ho, A_tr = gen_tr_ho_networks(A_orig, 0.8, 0.8)

        np.save(savepath + name + '_' + str(repeated) +'_Aho.npy', A_ho)
        np.save(savepath + name + '_' + str(repeated) +'_Atr.npy', A_tr)
        np.save(savepath + name + '_' + str(repeated) +'_Atest.npy', A_test)
        np.save(savepath + name + '_' + str(repeated) +'_Aorig.npy', A_orig)
        np.save(savepath + name + '_' + str(repeated) +'_Aval.npy', A_val)
        # train_tr, train_fl, test_tr, test_fl
        train_tr, ne_ho, e_diff, ne_orig = sample_true_false_edges(A_orig, A_tr, A_ho, repeated)

        #print(train_tr)

        nn, nnr, G1, G1_re, index_map, index_map_re = generate_graphs(train_tr, ne_ho) # generated trainning matrix
        nn_h, nnr_h, G1_h, G1_re_h, index_map_h, index_map_re_h = generate_graphs(e_diff, ne_orig) # generated testing matrix



        if len(G.nodes) > 20 and len(G.nodes)<1500 and (name not in do_not_touch) :

            for samp in sampling_methods:
                print(samp)


                start = time.time()            

                method_to_call = getattr(lbf, samp)
                sampler = method_to_call(nn)
                sampler_re = method_to_call(nnr)
                new_graph = sampler.sample(G1)
                reverse_graph = sampler_re.sample(G1_re)


                ng1 = nx.relabel_nodes(new_graph, index_map)
                ng1_re = nx.relabel_nodes(reverse_graph, index_map_re)


                Nsamples = 10000 # number of samples
                edge_t = [] # list of true edges (positive samples)
                edge_f = [] # list of false edges (negative samples)
                for ll in range(Nsamples):

                    #print(len(list(ng1.edges)))
                    edge_t_idx_aux = np.random.randint(len(list(ng1.edges)))
                    edge_f_idx_aux = np.random.randint(len(list(ng1_re.edges)))
                    #print(list(ng1.edges)[edge_t_idx_aux])
                    #print(list(ng1.edges)[edge_t_idx_aux][0])

                    edge_t.append((list(ng1.edges)[edge_t_idx_aux][0],list(ng1.edges)[edge_t_idx_aux][1]))
                    edge_f.append((list(ng1_re.edges)[edge_f_idx_aux][0],list(ng1_re.edges)[edge_f_idx_aux][1]))


                #DATA[samp+"_"+str(i) +"_"+ "t" + "_" + "train"][net] = ng1.edges()
                #DATA[samp+"_"+str(i) +"_"+ "f" + "_" + "train"][net] = ng1_re.edges()
                #DATA[samp+"_"+str(i) +"_"+ "t" + "_" + "train" + "_10000"][net] = edge_t
                #DATA[samp+"_"+str(i) +"_"+ "f" + "_" + "train" + "_10000"][net] = edge_f

                #np.savetxt(savepath + name + samp +"_"+ "t" + "_" + "train" + '.txt', ng1.edges())
                #np.savetxt(savepath + name + samp +"_"+ "f" + "_" + "train" + '.txt', ng1_re.edges())

                #np.savetxt(savepath + name + samp +"_"+ "t" + "_" + "train" + "_10000" + '.txt', edge_t)
                #np.savetxt(savepath + name + samp +"_"+ "f" + "_" + "train" + "_10000" + '.txt', edge_f)

                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "train" + '.npy', ng1.edges())
                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "f" + "_" + "train" + '.npy', ng1_re.edges())
                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "train" + "_10000" + '.npy', edge_t)
                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "f" + "_" + "train" + "_10000" + '.npy', edge_f)


                ############################################################################################

                #TEST CASE TEST CASE TEST CASE TEST CASE TEST CASE

                ############################################################################################


                #print("testcase")
                # Test Samples
                method_to_call = getattr(lbf, samp)
                sampler = method_to_call(nn_h)
                sampler_re = method_to_call(nnr_h)
                new_graph = sampler.sample(G1_h)
                reverse_graph = sampler_re.sample(G1_re_h)


                ng1 = nx.relabel_nodes(new_graph, index_map_h)
                ng1_re = nx.relabel_nodes(reverse_graph, index_map_re_h)

                Nsamples = 10000 # number of samples
                edge_t = [] # list of true edges (positive samples)
                edge_f = [] # list of false edges (negative samples)
                for ll in range(Nsamples):


                    edge_t_idx_aux = np.random.randint(len(list(ng1.edges)))
                    edge_f_idx_aux = np.random.randint(len(list(ng1_re.edges)))
                    edge_t.append((list(ng1.edges)[edge_t_idx_aux][0],list(ng1.edges)[edge_t_idx_aux][1]))
                    edge_f.append((list(ng1_re.edges)[edge_f_idx_aux][0],list(ng1_re.edges)[edge_f_idx_aux][1]))

                #DATA[samp+"_"+str(i) +"_"+ "t" + "_" + "test"][net] = ng1.edges()
                #DATA[samp+"_"+str(i) +"_"+ "f" + "_" + "test"][net] = ng1_re.edges()
                #DATA[samp+"_"+str(i) +"_"+ "t" + "_" + "test" + "_10000"][net] = edge_t
                #DATA[samp+"_"+str(i) +"_"+ "f" + "_" + "test" + "_10000"][net] = edge_f


                #np.savetxt(savepath + name + samp +"_"+ "t" + "_" + "test" + '.txt', ng1.edges())
                #np.savetxt(savepath + name + samp +"_"+ "f" + "_" + "test" + '.txt', ng1_re.edges())
                #np.savetxt(savepath + name + samp +"_"+ "t" + "_" + "test" + "_10000" + '.txt', edge_t)
                #np.savetxt(savepath + name + samp +"_"+ "f" + "_" + "test" + "_10000" + '.txt', edge_f)

                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "test" + '.npy', ng1.edges())
                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "f" + "_" + "test" + '.npy', ng1_re.edges())
                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "test" + "_10000" + '.npy', edge_t)
                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "f" + "_" + "test" + "_10000" + '.npy', edge_f)

        else:
            not_processed.append(net)
            with open('not_processed.txt', 'a') as fd:
                fd.write('\n')
                fd.write(str(net))
            np.savetxt('not_processed.txt', not_processed)

netlist = list(range(148,212)) 

with Pool(len(netlist)) as p:
    print(p.map(process_net, netlist))
