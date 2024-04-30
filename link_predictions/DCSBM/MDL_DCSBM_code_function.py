# +
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
import graph_tool.all as gt
import os
import time

def entropy_bin(q):
    if np.sum(q)==0:
        return 0
    q=q/float(np.sum(q))
    q_aux = np.array([0 if(qq==0) else qq*np.log(qq) for qq in q])
    return -1 * np.sum(q_aux)
def h_Lt(x):
    return (1+x)*np.log(1+x)-x*np.log(x)


def mdldcsbm_for_sampling(path, path1, net, samp):


    name = "net"+ str(net) +"_"
    test_name = "edge_set" +"_test"
    train_name = "edge_set" +"_train"

    edges_orig = np.loadtxt(path1 + name +'.txt').astype(int)
    train_edge = np.loadtxt(path + name + '_1' + samp +"_"+ "t" + "_" + "train" + '.npy').astype(int)
    num_nodes = int(np.max(edges_orig)) + 1
    
    method = "MDLDCSBM"
    tau = 0.01
    AUC = 0

    results_foldername = 'ResultsDCSBM_ST/'
    if not os.path.isdir(results_foldername):
        os.mkdir(results_foldername)

    start_time = time.time()  
    label_foldername = 'LabelMDLDCSBM_ST/'
    if not os.path.isdir(label_foldername):
        os.mkdir(label_foldername)    


    edges = np.loadtxt(path + name + '_1' + samp +"_"+ "t" + "_" + "train" + '.npy').astype(int)

    num_edges_info = len(edges) + 1
    
    edges = np.array(np.matrix(edges))
    
    
    row = np.array(edges)[:,0]
    col = np.array(edges)[:,1]
    
    data_aux = np.ones(len(row))
    Ao_aux = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    Ao_aux = sparse.triu(Ao_aux,1) + sparse.triu(Ao_aux,1).transpose()
    Ao_aux[Ao_aux>0] = 1          
          
    g_ds = gt.Graph(directed=False)
    v_g_ds={}
    e_g_ds={}
    for vv in range(num_nodes):
        v_g_ds[vv]=g_ds.add_vertex()
    if len(edges) != num_edges_info:
        print('error: inconsistency in the number of edges achieved by info file and the length of edge list')
    for ee in range(len(edges)):
        e_g_ds[ee] = g_ds.add_edge(edges[ee][0], edges[ee][1])    
    state = gt.minimize_blockmodel_dl(g_ds)
    B_inferred = state.get_B() #len(state.block_list)
    b_ds = state.b    

    label_inferred=[]
    try:
        for bb in b_ds:
            label_inferred.append(bb)  
    except:
        print('labels copied')
    label_inferred=np.array(label_inferred)
    label_inferred_stored=label_inferred.astype('int')
    
    np.savetxt(label_foldername+'label_'+ name + "_"+samp + "_" + method+'_f.txt',label_inferred_stored,fmt = '%d')
   
    
    ers = np.zeros((B_inferred,B_inferred))
    for bb1 in range(B_inferred):
        for bb2 in range(bb1,B_inferred):
            row_idx = np.where(label_inferred==bb1)[0]
            col_idx = np.where(label_inferred==bb2)[0]
            if bb1==bb2:
                ers[bb1,bb2] = np.sum(Ao_aux[row_idx[:,None],col_idx].toarray())
            else:
                ers[bb1,bb2] = np.sum(Ao_aux[row_idx[:,None],col_idx].toarray())
                ers[bb2,bb1] = ers[bb1,bb2]
    d_s = np.zeros((B_inferred,1))            
    for bb in range(B_inferred):
        d_s[bb] = np.sum(ers[bb,:])
    n_s = np.zeros((B_inferred,1))
    for bb in range(B_inferred):
        n_s[bb] = len(np.where(label_inferred==bb)[0])
        
    E_edges = len(edges)
    L_mclxty = E_edges*h_Lt(B_inferred*(B_inferred+1)/float(2*E_edges)) + num_nodes*np.log(B_inferred)
    
    deg_g = np.zeros((num_nodes,1))
    for nn in range(num_nodes):
        deg_g[nn] = np.sum(Ao_aux[nn,:].toarray())
    bins_deg_g = np.unique(deg_g)    
    Nk_deg_g_dict = {}
    pk_deg_g_dict = {}
    for nn in range(num_nodes):
        if deg_g[nn][0] in Nk_deg_g_dict:
            Nk_deg_g_dict[deg_g[nn][0]] += 1
            pk_deg_g_dict[deg_g[nn][0]] += 1/float(num_nodes)
        else:
            Nk_deg_g_dict[deg_g[nn][0]] = 1
            pk_deg_g_dict[deg_g[nn][0]] = 1/float(num_nodes)
    S_ent_dc = -E_edges
    for bb in bins_deg_g:
        S_ent_dc -= Nk_deg_g_dict[bb]*np.sum(np.log(np.array(range(1,int(bb+1)))))
    for bb1 in range(B_inferred):
        for bb2 in range(B_inferred):
            if ers[bb1,bb2]!=0:
                S_ent_dc -= 0.5*ers[bb1,bb2]*np.log(ers[bb1,bb2]/float(d_s[bb1]*d_s[bb2]))
    
    L_mclxty_dc = 1*L_mclxty
    for bb in bins_deg_g:
        if pk_deg_g_dict[bb]!=0:
            L_mclxty_dc -= num_nodes*pk_deg_g_dict[bb]*np.log(pk_deg_g_dict[bb])
    Q1 = S_ent_dc+L_mclxty_dc
    Q1_check = state.entropy()  
    #print("Q1 = ", Q1 , "Q1_check = ", Q1_check)
    del g_ds,state
#    row_orig = np.array(edges_orig)[:,0]
#    col_orig = np.array(edges_orig)[:,1]
#    
#    data_orig = np.ones(len(row_orig))
#    A_orig = csr_matrix((data_orig,(row_orig,col_orig)),shape=(num_nodes,num_nodes))
#    A_orig = A_orig + sparse.triu(A_orig,1).transpose()
#    A_orig[A_orig>0] = 1
#          
#    A_diff = A_orig - Ao_aux
#    e_diff = sparse.find(sparse.triu(A_diff,1))
#    
#    data_orig_aux = -1*np.ones(len(row_orig))
#    A_orig_aux = csr_matrix((data_orig_aux,(row_orig,col_orig)),shape=(num_nodes,num_nodes))
#    A_orig_aux = A_orig_aux + sparse.triu(A_orig_aux,1).transpose()
#    A_orig_aux= A_orig_aux + np.array([1])
#    ne_orig = sparse.find(sparse.triu(A_orig_aux,1))
    Nsamples = 10000;
    
    t_samples = np.loadtxt(path + name + "_1" + "_" + "t" + "_" + "test" + "_10000" + ".npy").astype('int')
    f_samples = np.loadtxt(path + name + "_1" + "_" + "f" + "_" + "test" + "_10000" + ".npy").astype('int')
    
    results = []
    TP_aux = 0
    for ll in range(Nsamples):
        if np.log2(ll) % 1 == 0:
            print("for ll = %s --- %s seconds ---" % (ll, (time.time() - start_time)))
#        edge_t_idx = np.random.randint(len(e_diff[0]))
#        edge_f_idx = np.random.randint(len(ne_orig[0]))
        
        edge_f = f_samples[ll]
        edge_t = t_samples[ll]
        edge_f = edge_f.astype('int')
        edge_t = edge_t.astype('int')

        E_edges_aux = E_edges+1
        L_mclxty_aux = E_edges_aux*h_Lt(B_inferred*(B_inferred+1)/float(2*E_edges_aux)) + num_nodes*np.log(B_inferred)
        pk_deg_g_aux = dict(pk_deg_g_dict)
        
        pk_deg_g_aux[deg_g[edge_t[0]][0]] -= 1/float(num_nodes)
        pk_deg_g_aux[deg_g[edge_t[1]][0]] -= 1/float(num_nodes)
        if deg_g[edge_t[0]][0]+1 in pk_deg_g_aux:
            pk_deg_g_aux[deg_g[edge_t[0]][0]+1] += 1/float(num_nodes)
        else:
            pk_deg_g_aux[deg_g[edge_t[0]][0]+1] = 1/float(num_nodes)
        if deg_g[edge_t[1]][0]+1 in pk_deg_g_aux:
            pk_deg_g_aux[deg_g[edge_t[1]][0]+1] += 1/float(num_nodes)
        else:
            pk_deg_g_aux[deg_g[edge_t[1]][0]+1] = 1/float(num_nodes)
        bb1 = label_inferred[edge_t[0]]
        bb2 = label_inferred[edge_t[1]]
        S_ent_dc_aux = S_ent_dc - 1 #-1 for added edge
        S_ent_dc_aux += np.sum(np.log(np.array(range(1,int(deg_g[edge_t[0]]+1)))))
        S_ent_dc_aux += np.sum(np.log(np.array(range(1,int(deg_g[edge_t[1]]+1)))))
        S_ent_dc_aux -= np.sum(np.log(np.array(range(1,int(deg_g[edge_t[0]]+2)))))
        S_ent_dc_aux -= np.sum(np.log(np.array(range(1,int(deg_g[edge_t[1]]+2)))))
        if bb1==bb2:
            if ers[bb1,bb2]!=0:
                S_ent_dc_aux += 0.5*ers[bb1,bb2]*np.log(ers[bb1,bb2]/float(d_s[bb1]*d_s[bb2]))
        else:
            if ers[bb1,bb2]!=0:
                S_ent_dc_aux += 0.5*ers[bb1,bb2]*np.log(ers[bb1,bb2]/float(d_s[bb1]*d_s[bb2]))
                S_ent_dc_aux += 0.5*ers[bb2,bb1]*np.log(ers[bb2,bb1]/float(d_s[bb2]*d_s[bb1]))
        if bb1==bb2:
                S_ent_dc_aux -= 0.5*(ers[bb1,bb2]+2)*np.log((ers[bb1,bb2]+2)/float((d_s[bb1]+2)*(d_s[bb2]+2)))    
        else:
            S_ent_dc_aux -= 0.5*(ers[bb1,bb2]+1)*np.log((ers[bb1,bb2]+1)/float((d_s[bb1]+1)*(d_s[bb2]+1)))
            S_ent_dc_aux -= 0.5*(ers[bb2,bb1]+1)*np.log((ers[bb2,bb1]+1)/float((d_s[bb2]+1)*(d_s[bb1]+1)))
        
        L_mclxty_dc_aux = 1*L_mclxty_aux
        deg_g_aux = 1*deg_g    
        deg_g_aux[edge_t[0]] += 1
        deg_g_aux[edge_t[1]] += 1
        bins_deg_g_aux = np.unique(deg_g_aux)
        for bb in bins_deg_g_aux:
            if pk_deg_g_aux[bb]!=0:
                L_mclxty_dc_aux -= num_nodes*pk_deg_g_aux[bb]*np.log(pk_deg_g_aux[bb])
        Q_t = S_ent_dc_aux+L_mclxty_dc_aux  

        E_edges_aux = E_edges+1
        L_mclxty_aux = E_edges_aux*h_Lt(B_inferred*(B_inferred+1)/float(2*E_edges_aux)) + num_nodes*np.log(B_inferred)
        pk_deg_g_aux = dict(pk_deg_g_dict)
        
        pk_deg_g_aux[deg_g[edge_f[0]][0]] -= 1/float(num_nodes)
        pk_deg_g_aux[deg_g[edge_f[1]][0]] -= 1/float(num_nodes)
        if deg_g[edge_f[0]][0]+1 in pk_deg_g_aux:
            pk_deg_g_aux[deg_g[edge_f[0]][0]+1] += 1/float(num_nodes)
        else:
            pk_deg_g_aux[deg_g[edge_f[0]][0]+1] = 1/float(num_nodes)
        if deg_g[edge_f[1]][0]+1 in pk_deg_g_aux:
            pk_deg_g_aux[deg_g[edge_f[1]][0]+1] += 1/float(num_nodes)
        else:
            pk_deg_g_aux[deg_g[edge_f[1]][0]+1] = 1/float(num_nodes)
        bb1 = label_inferred[edge_f[0]]
        bb2 = label_inferred[edge_f[1]]
        S_ent_dc_aux = S_ent_dc - 1 #-1 for added edge
        S_ent_dc_aux += np.sum(np.log(np.array(range(1,int(deg_g[edge_f[0]]+1)))))
        S_ent_dc_aux += np.sum(np.log(np.array(range(1,int(deg_g[edge_f[1]]+1)))))
        S_ent_dc_aux -= np.sum(np.log(np.array(range(1,int(deg_g[edge_f[0]]+2)))))
        S_ent_dc_aux -= np.sum(np.log(np.array(range(1,int(deg_g[edge_f[1]]+2)))))
        if bb1==bb2:
            if ers[bb1,bb2]!=0:
                S_ent_dc_aux += 0.5*ers[bb1,bb2]*np.log(ers[bb1,bb2]/float(d_s[bb1]*d_s[bb2]))
        else:
            if ers[bb1,bb2]!=0:
                S_ent_dc_aux += 0.5*ers[bb1,bb2]*np.log(ers[bb1,bb2]/float(d_s[bb1]*d_s[bb2]))
                S_ent_dc_aux += 0.5*ers[bb2,bb1]*np.log(ers[bb2,bb1]/float(d_s[bb2]*d_s[bb1]))
        if bb1==bb2:
                S_ent_dc_aux -= 0.5*(ers[bb1,bb2]+2)*np.log((ers[bb1,bb2]+2)/float((d_s[bb1]+2)*(d_s[bb2]+2)))    
        else:
            S_ent_dc_aux -= 0.5*(ers[bb1,bb2]+1)*np.log((ers[bb1,bb2]+1)/float((d_s[bb1]+1)*(d_s[bb2]+1)))
            S_ent_dc_aux -= 0.5*(ers[bb2,bb1]+1)*np.log((ers[bb2,bb1]+1)/float((d_s[bb2]+1)*(d_s[bb1]+1)))
        
        L_mclxty_dc_aux = 1*L_mclxty_aux
        deg_g_aux = 1*deg_g    
        deg_g_aux[edge_f[0]] += 1
        deg_g_aux[edge_f[1]] += 1
        bins_deg_g_aux = np.unique(deg_g_aux)
        for bb in bins_deg_g_aux:
            if bb in list(pk_deg_g_aux.keys()) and pk_deg_g_aux[bb]!=0:
                L_mclxty_dc_aux -= num_nodes*pk_deg_g_aux[bb]*np.log(pk_deg_g_aux[bb])
        Q_f = S_ent_dc_aux+L_mclxty_dc_aux
        if Q_t < Q_f:
            TP_aux += 1
            results.append((list(edge_t)+list(edge_f)+[Q_t,Q_f,1]))
        elif Q_t == Q_f:
            if np.random.randint(2)==0:
                TP_aux +=1
                results.append((list(edge_t)+list(edge_f)+[Q_t,Q_f,1]))
            else:
                results.append((list(edge_t)+list(edge_f)+[Q_t,Q_f,0]))
        else:
            results.append((list(edge_t)+list(edge_f)+[Q_t,Q_f,0]))
    print('observed fraction of edges for ', name + "_" + samp, '% is done.')       
    AUC = TP_aux/float(Nsamples)
    if net == 0:
        results_foldername = 'ResultsDCSBM_ST/'
        if not os.path.isdir(results_foldername):
            os.mkdir(results_foldername)
            
        fid = open(results_foldername + 'AUC_MDLDCSBM_'+ name + "_" + samp +'.txt','w')
        fid.write(str(AUC)+'\n')
        fid.close()
    else:
        results_foldername = 'ResultsDCSBM_ST/'
        fid = open(results_foldername + 'AUC_MDLDCSBM_'+ name + "_" + samp  +'.txt','a')
        fid.write(str(AUC)+'\n')
        fid.close()
    results = np.array(results)
    
    np.savetxt(results_foldername + "resultsMDLDCSBM_"+ name + "_" + samp + ".txt",results,fmt='%u %u %u %u %f %f %u')
    
    return AUC


# -


