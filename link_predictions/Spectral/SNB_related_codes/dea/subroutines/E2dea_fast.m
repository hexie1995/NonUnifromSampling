function [ dea,deaM, deaIM ] = E2dea_fast( E )
%Compute DEA matrix and marginal-edge matrix given edge list E
%   usage: [T,M]=E2dea_fast(E)
%   this function uses loops, so it is slower than c++ version
    if(nargin ~= 1)
        fprintf('[T,M]=E2dea(E)');
        dea=0; deaM=0;
        return;
    end
    %tic
    %% compute N, M, Aij
    E1=E(:,1);
    E2=E(:,2);
    %size(E)
    M=length(E);
    N=max(max(E1),max(E2));
    A = sparse(E1,E2,ones(M,1),N,N,M);
    A = A+A'; % Adjacency matrix
    A(A>1.1)=1; %ignor self edges.

    nodes=[E1 E2;E2 E1];
    nodes1=nodes(:,1);%first node of edges
    nodes2=nodes(:,2);%second node of edges  nodes1 -> nodes2
    A2edge = A; % id of edges
    tM=2*M;% two times M
    for a=1:tM
        A2edge(nodes1(a),nodes2(a))=a;
        %disp([nodes1(a) nodes2(a) a]);
    end
    %% compute dea, deaM
    %A(A>1.1)=1; %ignor self edges.
    %A
    di=sum(A);
    di2=di.*(di-1);
    maxdi=max(di);
    es=zeros(maxdi,2);
    dM=sum(di2); % number of edges of dea matrix, which is \sum_i d_i(d_i-1)
    dMM=sum(di); % number of edges of deaM matrix, which is \sum_i d_i
    deaE=ones(dM,2);   % Edge list of dea matrix
    deaEM=ones(dMM,2); % Edge list of deaM matrix
    deaEIM=ones(dM,2); % Edge list of deaIM matrix
    eidx=0; emidx=0;
    dM_est=0;
    for i=1:N % for each node, find its neighbors
        idxi=find(A(i,:));
        di=numel(idxi);
        for idxj=1:di
            j=idxi(idxj); 
            aij=A2edge(i,j);%edge id of i to j
            aji=A2edge(j,i);%edge id of j to i
            es(idxj,1)=aij; % i to j
            es(idxj,2)=aji; % j to i
        end
        for idxj=1:di;
            emidx=emidx+1;
            deaEM(emidx,1)=i; % i
            deaEM(emidx,2)=es(idxj,2); % j to i
            for idxk=1:di;
                if(idxj==idxk) 
                    continue;
                end
                k=idxi(idxk);
                eidx = eidx+1;
                deaE(eidx,1)=es(idxj,1); % i to j
                deaE(eidx,2)=es(idxk,2); % k to i
                dM_est=dM_est+1;
                deaEIM(eidx,1)=es(idxj,1); % i to j
                deaEIM(eidx,2)=k; % k
            end
        end
    end
    
    if(dM ~= dM_est)
        fprintf('Be careful, here may be multi-loops!');
    end
    dea=sparse(deaE(:,1),deaE(:,2),ones(dM,1),tM,tM,dM);
    %sum(sum(dea)<2)
    deaM=sparse(deaEM(:,1),deaEM(:,2),ones(dMM,1),N,tM,dMM);
    %fprintf('Average degree of dog matrix is : %g\n',full(sum(sum(dea))/length(dea)));
    deaIM=sparse(deaEIM(:,1),deaEIM(:,2),ones(dM,1),tM,N,dM);
    %deaE
    %dea
    %toc
end