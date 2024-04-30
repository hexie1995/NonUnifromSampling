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
loadpath = r"/home/xhe/updated_edges_revised//"
savepath = r"/home/xhe/sampled_edges_2024//"
EDGELIST = DATA["edges_id"]

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


# +
def help_create_matrix(edges_orig, num_nodes):
    
    #num_nodes = int(np.max(edges_orig)) + 1
    row = np.array(edges_orig)[:,0]
    col = np.array(edges_orig)[:,1]

    data_aux = np.ones(len(row))
    A_orig = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A_orig = sparse.triu(A_orig,1) + sparse.triu(A_orig,1).transpose()
    A_orig[A_orig>0] = 1 
    A_orig = A_orig.todense()
    
    return A_orig

def help_sample_edges(edges):
    N_samples = 10000
    edge_count = len(list(edges))
    sampled_edge = []
    for ll in range(N_samples):
        idx = np.random.randint(edge_count)  
        sampled_edge.append((list(edges)[idx][0],list(edges)[idx][1]))
    return sampled_edge


# +
def after_process_net(net):
    ############################## COMMENTS, READ CAREFULLY####################################
    # total network is already saved in the loadpath, do not process again.
    # But we still need to load the edgelist first just for comparison later.
    # We set the name first. 
    ############################## COMMENTS, READ CAREFULLY####################################
    
    num_nodes = DATA["number_nodes"][net]
    name = "net"+ str(net) + "_"
    if num_nodes> 20 and num_nodes <1500 and (name not in do_not_touch):
    
    #     edges_orig = EDGELIST[net]
    #     np.savetxt(savepath + name +'.txt', edges_orig)

        for repeated in range(1,6):
            ############################## COMMENTS, READ CAREFULLY####################################
            # A_orig and A_test are already generated and we do NOT generate them again to save process time.
            # They can all be found in load_path, we directly load them. 
            ############################## COMMENTS, READ CAREFULLY####################################
            A_ho = np.load(loadpath + name + '_' + str(repeated) +'_Aho.npy')
            A_orig = np.load(loadpath + name + '_' + str(repeated) +'_Aorig.npy')
            A_test = np.load(loadpath + name + '_' + str(repeated) +'_Atest.npy') 

            G_test = nx.from_numpy_matrix(A_test)

            # test negative edges, because they are the same within e run for all sampling methods.
            A_orig_neg = -1*A_orig + 1
            G_test_neg = nx.from_numpy_matrix(A_orig_neg)

            # train negative edges + valid negative edges, they are from the same set, we just sample twice. 
            A_ho_neg = -1*A_ho + 1 
            G_train_neg = nx.from_numpy_matrix(A_orig_neg)

            test_pos = help_sample_edges(G_test.edges())
            test_neg = help_sample_edges(G_test_neg.edges())
            
            #holdout_neg = help_sample_edges(G_train_neg.edges())

            train_neg_10000 = help_sample_edges(G_train_neg.edges())
            valid_neg_10000 = help_sample_edges(G_train_neg.edges())

            ############################## COMMENTS, READ CAREFULLY####################################
            # Very importantly, both the test edges and the negative edges are not affected by which sampling method. 
            ############################## COMMENTS, READ CAREFULLY####################################

            np.savetxt(savepath + name + '_' + str(repeated) +"_"+ "t" + "_" + "test" + "_10000" + '.npy', test_pos)
            np.savetxt(savepath + name + '_' + str(repeated) +"_"+ "f" + "_" + "test" + "_10000" + '.npy', test_neg)
            np.savetxt(savepath + name + '_' + str(repeated) +"_"+ "f" + "_" + "train" + "_10000" + '.npy', train_neg_10000)
            np.savetxt(savepath + name + '_' + str(repeated) +"_"+ "f" + "_" + "valid" + "_10000" + '.npy', valid_neg_10000)

#             np.savetxt(savepath + name + '_' + str(repeated) +"_"+ "t" + "_" + "test" + '.npy', G_test.edges())
#             np.savetxt(savepath + name + '_' + str(repeated) +"_"+ "f" + "_" + "test" + '.npy', G_test_neg.edges())
#             np.savetxt(savepath + name + '_' + str(repeated) +"_"+ "f" + "_" + "train" + '.npy', G_train_neg.edges())
        
            
            
            np.save(savepath + name + '_' + str(repeated) +'_Aho.npy', A_ho)
            np.save(savepath + name + '_' + str(repeated) +'_Atest.npy', A_test)
            np.save(savepath + name + '_' + str(repeated) +'_Aorig.npy', A_orig)


            ############################## COMMENTS, READ CAREFULLY####################################
            # Note that ng1.edges() is A_tr in the pipeline, which is samp(LCC(A_ho))
            # So We can get A_orig and A_test directly by loading.
            # We get A_tr by loading loadpath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "train" + '.npy'
            # We then get A_ho = A_orig - A_test
            # We get A_valid = A_ho - A_tr
            # This way we do not need to run anything at all. 
            # We then save A_tr, A_ho, A_orig, A_test to the new savepath for future reference.
            # And then we also random sample 10000 positive edges from A_tr, A_valid, A_test.
            ############################## COMMENTS, READ CAREFULLY####################################

            for samp in sampling_methods:
                print(samp)
                A_tr_edges = np.loadtxt(loadpath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "train" + '.npy')

                A_tr_samp = help_create_matrix(A_tr_edges, A_orig.shape[0])
                A_valid_samp = A_ho - A_tr_samp

                G_valid = nx.from_numpy_matrix(A_valid_samp)

                train_pos = np.loadtxt(loadpath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "train" + "_10000" + '.npy')
                valid_pos = help_sample_edges(G_valid.edges())
                
                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "train" + '.npy', A_tr_edges)
                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "train_10000" + '.npy', train_pos)
                np.savetxt(savepath + name + '_' + str(repeated) + samp +"_"+ "t" + "_" + "valid_10000" + '.npy', valid_pos)
                np.save(savepath + name + '_' + str(repeated) + samp +'_Atr.npy', A_tr_samp)
                np.save(savepath + name + '_' + str(repeated) + samp +'_Aval.npy', A_valid_samp)

                ############################## COMMENTS, READ CAREFULLY####################################
                # Everything in the test case needs to be rerun because all of our edges were sampled, but we should not sample.
                # We should just directly take the test set, A_test and just sample 10000 true edges and false edges directly.
                ############################## COMMENTS, READ CAREFULLY####################################

    else:
        with open('not_processed.txt', 'a') as fd:
            fd.write('\n')
            fd.write(str(net))
# -

netlist = list(range(0,150))
after_process_net(1)

a = {"a":1, "b":2}
list(a.keys())

# with Pool(len(netlist)) as p:
#      print(p.map(after_process_net, netlist))

