# +
import inspect
import tempfile
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
import os
import sys
from shutil import copyfile
from subprocess import Popen
#sys.path.append('/Users/amgh5286/stacking/new_real_data/ST_algs/LD/Infomap/Infomap/examples/python/')
import infomap



def infomap_for_sampling(path, net, samp):


    name = "net"+ str(net) +"_"
    test_name = "edge_set" +"_test"
    train_name = "edge_set" +"_train"

    edges_orig = np.loadtxt(path + name +'.txt')
    train_edge = np.loadtxt(path + name + train_name +'.txt')
    num_nodes = int(np.max(train_edge)) + 1

    tau = 0.01
    AUC = 0


    method = 'Infomap'
    label_foldername = './label_'+method+'/'
    if not os.path.isdir(label_foldername):
        os.mkdir(label_foldername)  
    results_foldername = './ResultsLPST_Infomap/'
    if not os.path.isdir(results_foldername):
        os.mkdir(results_foldername)


        
    path_EL_sampled = path + name + train_name  +".txt"

    edges = np.loadtxt(path_EL_sampled)
    num_edges_info = len(edges)
    
    edges = np.array(np.matrix(edges))
    
    fid = open(path_EL_sampled,'r')
    lines = fid.readlines()
    
    
    #n = num_nodes = int(lines[1][lines[1].find('# # of nodes:')+14::].strip())
   #num_edges_info = int(lines[4][lines[4].find('# number of edges revealed:')+28::].strip())
    
    stateNetwork = """*Vertices """ + str(num_nodes) + """\n"""
    for nn in range(1,num_nodes+1):
        stateNetwork = stateNetwork + str(nn) + """ """ + '"' +str(nn)+ '"' +"""\n"""
    stateNetwork = stateNetwork + """*Edges """ +str(num_edges_info) + """\n"""
    for ee in edges:
        stateNetwork = stateNetwork + str(int(ee[0])+1) + """ """ + str(int(ee[1])+1) +"""\n"""
        # stateNetwork = stateNetwork + str(int(ee[1])+1) + """ """ + str(int(ee[0])+1) +"""\n"""
    str_fname = "stateNetwork_n_" + name +"_"+ samp + ".net"
    filename_aux = "%s/%s" % (tempfile.gettempdir(),str_fname)
    with open(filename_aux, 'w') as fp:
        fp.write(stateNetwork)
    print("Wrote state network to file '{}'".format(filename_aux))

    infomapWrapper = infomap.Infomap("--two-level")
    infomapWrapper.readInputData(filename_aux)
    infomapWrapper.run()
    tree = infomapWrapper.tree
    #inspect.getmembers(infomap.InfoNode, lambda a:not(inspect.isroutine(a)))
    Q1 = tree.codelength
    node_phys_ind = [] 
    node_module_ind = []
    print(tree)
    for node in infomapWrapper.iterLeafNodes():
    #for node in tree.leafIter():
        #print(node.moduleIndex())
        #node_phys_ind.append(node.physIndex)
        node_phys_ind.append(node.index)
        node_module_ind.append(node.moduleIndex())
    node_phys_ind = np.array(node_phys_ind)
    node_module_ind = np.array(node_module_ind)
    node_sort_phys_ind = np.argsort(node_phys_ind)
    label_inferred = node_module_ind[node_sort_phys_ind]
    np.savetxt(label_foldername+'label_'+ name + "_"+samp + "_" + method+'_f'+'.txt',label_inferred,fmt = '%d')
    
    method = 'Infomap' 
    label_method = label_inferred
    bestK_method = len(np.unique(label_method))
    
    unique_types = np.unique(label_method)
    unique_types_mapped = range(len(unique_types))
    unique_types_mapped = {}
    for ll in range(len(unique_types)):
        unique_types_mapped[unique_types[ll]] = ll
    for ll in range(len(label_method)):
        label_method[ll] = unique_types_mapped[label_method[ll]]

    cluNetwork = """# file_n_"""  + """\n"""
    cluNetwork = cluNetwork + "# node cluster \n"
    for nn in range(1,num_nodes+1):
        cluNetwork = cluNetwork + str(nn) + """ """ +str(label_method[nn-1])+ """\n"""
    clu_fname = "cluNetwork_n_" +".clu"
    clufilename_aux = "%s/%s" % (tempfile.gettempdir(),clu_fname)
    with open(clufilename_aux, 'w') as fp:
        fp.write(cluNetwork)


    Nsamples = 10000;
    
    t_samples = np.loadtxt(path + name +  "_2" + samp +"_"+ "t" + "_" + "train" + "_10000" + ".npy").astype('int')
    f_samples = np.loadtxt(path + name +  "_2" + samp +"_"+ "f" + "_" + "train" + "_10000" + ".npy").astype('int')
    
    results = []
    TP_aux = 0
    
    for ll in range(Nsamples):
        edge_f = f_samples[ll]
        edge_t = t_samples[ll]
        
        stateNetwork_aux_t = stateNetwork + str(int(edge_t[0])+1) + """ """ + str(int(edge_t[1])+1) +"""\n"""
        str_fname_t = "LDstateNetwork_aux_n_" + name +"_"+ samp +"_t.net"
        filename_aux_t = "%s/%s" % (tempfile.gettempdir(),str_fname_t)
        with open(filename_aux_t, 'w') as fp:
            fp.write(stateNetwork_aux_t)
        infomapWrapper_aux_t = infomap.Infomap("--no-infomap -c "+clufilename_aux)
        infomapWrapper_aux_t.readInputData(filename_aux_t)
        infomapWrapper_aux_t.run()
        Q_t = infomapWrapper_aux_t.tree.codelength
        del infomapWrapper_aux_t
        dQ_t = Q1-Q_t

        stateNetwork_aux_f = stateNetwork + str(int(edge_f[0])+1) + """ """ + str(int(edge_f[1])+1) +"""\n"""
        str_fname_f = "LDstateNetwork_aux_n_"+ name +"_"+ samp  +"_f.net"
        filename_aux_f = "%s/%s" % (tempfile.gettempdir(),str_fname_f)
        with open(filename_aux_f, 'w') as fp:
            fp.write(stateNetwork_aux_f)
        infomapWrapper_aux_f = infomap.Infomap("--no-infomap -c "+clufilename_aux)
        infomapWrapper_aux_f.readInputData(filename_aux_f)
        infomapWrapper_aux_f.run()
        Q_f = infomapWrapper_aux_f.tree.codelength
        del infomapWrapper_aux_f
        dQ_f = Q1-Q_f
        
        if dQ_t > dQ_f:
            TP_aux += 1
            results.append((list(edge_t)+list(edge_f)+[Q_t,Q_f,1]))
        elif dQ_t == dQ_f:
            if np.random.randint(2)==0:
                TP_aux +=1
                results.append((list(edge_t)+list(edge_f)+[Q_t,Q_f,1]))
            else:
                results.append((list(edge_t)+list(edge_f)+[Q_t,Q_f,0]))
        else:
            results.append((list(edge_t)+list(edge_f)+[Q_t,Q_f,0]))

    print ('observed fraction of edges for ', name + "_" + samp ,'% is done.')         
    AUC = TP_aux/float(Nsamples)
    
    if net == 0:
        results_foldername = './ResultsLPST_Infomap/'
        if not os.path.isdir(results_foldername):
            os.mkdir(results_foldername)
        fid = open(results_foldername + 'AUC_Infomap_'+ name + "_" + samp +'.txt','w')
        fid.write(str(AUC)+'\n')
        fid.close()
    else:
        results_foldername = './ResultsLPST_Infomap/'
        fid = open(results_foldername + 'AUC_Infomap_'+ name + "_" + samp +'.txt','a')
        fid.write(str(AUC)+'\n')
        fid.close()

    results = np.array(results)
    np.savetxt(results_foldername + "resultsInfomap_"+ name + "_" + samp +".txt",results,fmt='%u %u %u %u %f %f %u')
    
    return AUC
