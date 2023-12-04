function [ dea,deaM ] = E2dea( E )
%Compute DEA matrix and marginal-edge matrix given edge list E
%   usage: [T,M]=E2dea(A)
%   this function uses loops, so it is much slower than c++ version
%   this is a slow version, and will be not used in main function.
    if(nargin ~= 1)
        fprintf('[T,M]=E2dea(A)');
        dea=0; deaM=0;
        return;
    end
    tic
    %% compute N, M, Aij
    E1=E(:,1);
    E2=E(:,2);
    %size(E)
    M=length(E);
    N=max(max(E1),max(E2));
    A = sparse(E1,E2,ones(M,1),N,N,M);
    A = A+A'; % Adjacency matrix

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
    dea=sparse(1,1,0,tM,tM); % Dea matrix with dimension [2M, 2M]
    deaM=sparse(1,1,0,N,tM); % Matrix for computing marginals
    for i=1:N % for each node, find its neighbors
        idxi=find(A(i,:));
        for idxj=1:numel(idxi)
            j=idxi(idxj); 
            aij=A2edge(i,j);%edge id of i to j
            aji=A2edge(j,i);%edge id of j to i
            deaM(i,aji)=1;
            for idxk=1:numel(idxi)
                if(idxk == idxj)
                    continue
                end
                k=idxi(idxk); %i to k, there must be k to i
                aki=A2edge(k,i);%edge index of k to i
                dea(aij,aki)=1;
            end
        end
    end
    toc
    %dea
end