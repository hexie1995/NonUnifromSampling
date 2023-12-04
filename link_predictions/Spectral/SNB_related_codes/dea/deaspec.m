function [result] = deaspec(varargin)
% DEASPEC do spectral clustering using various matrices including DEA matrix.
%
% Usage:  result =deaspec(parameters)
%    parameters can be in one of two forms
%    1.) A struct containing fields. For example: 
%    opt.N=100; opt.c=5; result = deaspec(opt);
%
%    2.) Pair of arguments given in following way:
%    result = deaspec('N',100,'c',5);
%
% Return values: a struct containing following fields:
%    D:             eigenvalues
%    V:             eigenvectors
%    sigma:         infered configuration
%    ovl:           overlap.
%    dea:           DEA matrix of the graph
%    E:             Edge list of the graph
%    c_dea:         average degree of DEA matrix
%    conf_infer:    infered configuration
%    conf_true:     true configuration
%    Dsort:         eigenvalues corresponding to eigenvectors that used
%                   in clustering
%    Vsort:         eigenvectors that used in clustering
%    cuts:          cut size given by the infered configuration
%    rcuts:         ratio cut size given by the infered configuration
%    ncuts:          normalized cut size given by the infered configuration
% PARAMETERS:
%    N, number of elements, default: 100
%    c, average degree, default: 3
%    q, number of groups, default: 2
%    mode, select a matrix to do spectral clustering, default: 5
%        0: DEA matrix
%        1: Modularity matrix
%        2: random walk matrix 
%        3: Adjacency matrix
%        4: Normalized Laplacian. 
%        5: Matrix F, called B' in the paper, which has the same set of
%        non-trivial engenvalues and eigenvectors as DEA matrix.
%    lcc, flat to use larest connected component, default: 0
%        1: use LCC
%        0: do not use LCC.
%    seed, random seed used to generate a random graph, default:1
%    epsilon, c_out/c_in, default: 0.1 
%    Note that is epsilon=1, it gives a random or random regular graph.
%    gen_flag, a flag to generate a network, default: 1 
%        0: generate the network and  compute DEA matrix by SBM. 
%        1: generate a random network and compute DEA matrix by MATLAB code. 
%        2: generate a regular random network by rewiring.
%        3: generate a regular random network by configuration model.
%        4: read adjacency and DEA matrices from file.
%    basename, name of file to read or write, default: 'test'.
%    gmlfname, write the generated or loaded network to a gml file.
%    ovl_real, use only real eigenvalues to compute overlap.
%    ftype, type to open file
%        'gml': gml
%        'spm': edge list
%        'nl' : node list
%    overflag, compute overlap to true configuration, default: 0
%        1: compute overlap
%        0: do not compute overlap
%    approx, approximate eigenvector that correlated with planted
%    configuration, default: 0
%        0: do not use approximate eigenvector.
%        1: use approximate eigenvector instead of 2nd eigenvector of
%        matrix.
%    do_clustering, compute overlap by doing clustering, default: 0
%        0: infer configuration using sign of eigenvector. There will be one configuration for each
%        eigenvector from cbegin's to cend's.
%        1: infer configuration by grouping elements of one eigenvector
%        using information of true group size.
%        2: infer configuration by appying k-means algorithm on a matrix
%        containing eigenvectors from cbegin's to cend's as columns.
%        3: as 2, but using k-means on the matrix that each row is
%        normalized.
%        4: as 2, but using k-means on the matrix that basis is changed by
%        Principle Component Analysis.
%

%% command line parameters

N=100; q=2; c=3; mode=5; gen_flag=1; numvec=2; numvec_eigs=6; lcc=0; seed=1; epsilon=0.1; basename='test'; do_clustering=0; cbegin=1; cend=-1; ovl_real=1;
D=[]; V=[]; conf_infer=[]; ovl=0; ftype=''; approx=0; gmlfname='';ovl_norm=0;
c_dea=0;
%celldisp(varargin)
if((nargin==0) );
    fprintf('Usage: deaspec(parameters).\n');
    fprintf('Try');            cprintf('blue',' help deaspec\n');
    return;
end
if(nargin==1)
    opts=varargin{1};
    if(~isstruct(opts))
        cprintf('red','If only one argument is given, it must be given as a struct.\n');
        fprintf('Try');            cprintf('blue',' help deaspec\n'); return
    end
    fns=fieldnames(opts);
    for idx=1:numel(fns);
        i=fns{idx};
        if(i=='n')
            i='N';
        end
        j=getfield(opts,i);
        if(ischar(j));
            txt=sprintf('%s=%s;',i,j);
        elseif( (isinteger(j) || isfloat(j)) && length(j)==1)
            txt=sprintf('%s=%g;',i,j);
        else
            cprintf('red', 'Wrong arguments, parameters must be a string, integer or float\n');
            fprintf('Try');        cprintf('blue',' help deaspec\n');            return;
        end
        if(exist(i)==1)
            eval(txt);
        else
            cprintf('red', 'Variable %s does not exist!\n', i);
            fprintf('Try');        cprintf('blue',' help deaspec\n');            return;
        end
    end
else
    if(mod(numel(varargin),2)~=0)
        fprintf('If more than 2 parameters are given, they must be given in pairs\n');
        fprintf('Try');            cprintf('blue',' help deaspec\n'); return
    end
    for idx=1:2:numel(varargin)
        i=varargin{idx};
        if(i=='n')
            i='N';
        end
        if ( ~ischar(i))
            cprintf('red','Wrong arguments, parameter %g is supposed to be a stirng\n',i);
            fprintf('Try');        cprintf('blue',' help deaspec\n');            return;
        end
        j=varargin{idx+1};
        if(ischar(j));
            txt=sprintf('%s=''%s'';',i,j);
        elseif( (isinteger(j) || isfloat(j)) && length(j)==1)
            txt=sprintf('%s=%g;',i,j);
        else
            cprintf('red', 'Wrong arguments, parameters must be a string, integer or float\n');
            fprintf('Try');        cprintf('blue',' help deaspec\n');            return;
        end
        if(exist(i,'var')==1)
            eval(txt);
        else
            cprintf('red', 'Variable %s does not exist!\n', i);
            fprintf('Try');        cprintf('blue',' help deaspec\n');            return;
        end
    end
end
if(cend<0) %cend not set, so set a default value to it.
    cend=abs(numvec);
end


    %% load or generate a network
    if( ~strcmp(ftype,'') )
        fprintf('loading %s file %s \n',ftype,basename);
        [E, A, conf_true] = load_file(ftype,basename,lcc);
    else
        [E, A, conf_true]=gen_network(gen_flag, N, c, epsilon, q, seed, basename, lcc);
    end
    fprintf('N=%d, M=%d, q=%d\n',max(max(E)),length(E),max(conf_true));
    %% write to gmal file
    if( ~strcmp(gmlfname,'') )
        write_gml(gmlfname, A, E, conf_true);
    end
    
    %% compute dea matrix
    DEA=0; DEAM=0; DEAIM=0;
    if(mode == 0) % using DEA matrix
        [ DEA,DEAM,DEAIM ] = get_dea( gen_flag, E, basename );
        %[ DEA,DEAM ] = get_dea_fast( gen_flag, E, basename );
        c_dea=full(sum(sum(DEA))/length(DEA));
        fprintf('Average degree of DEA matrix is : %g\n',c_dea);

    end
    tic
    if(approx == 0 )
        %% find spectrum
        [ V,D ] = get_spectrum( A, DEA, mode, numvec, numvec_eigs);
    else
        %% use approximate eigenvector
        [ V,D ] = get_approx_vec(A, DEA, DEAIM, mode, approx, conf_true);
    end
    toc
    %% infer configuration
    ovl=0; conf_infer=[]; Dsort=[]; Vsort=[]; cuts=[];Rcuts=[];Ncuts=[];
    if(do_clustering>=0)
        [Dsort,conf_infer,Vsort]=compute_conf_infer(q,D,V,numvec,do_clustering,mode,cbegin,cend,ovl_real,0,conf_true,DEAM);
        [cuts Rcuts Ncuts]=get_cuts(E,conf_infer);
    end
    %[conf_true,conf_infer]
    
    %% compute overlap
    if(~(isempty(conf_true)) && (max(conf_true) == max(max(conf_infer))))
        ovl=get_ovl(conf_true,conf_infer,ovl_norm);
    end
    
    %% remove temperary files
    if(gen_flag==0) % network is generated by SBM, so let's remove the temperary files.
        rm_files(basename);
    end
    
    result.D=D;
    result.V=V;
    result.ovl=ovl;
    result.conf_infer=conf_infer;
    result.conf_true=conf_true;
    result.Dsort=Dsort;
    result.Vsort=Vsort;
    result.cuts=cuts;
    result.ncuts=Ncuts;
    result.rcuts=Rcuts;
    result.c_dea=c_dea;
    result.E=E;
    result.dea=DEA;
          
end