function [ A ] = E2A( E )
%E2A compute adjacency matrix from edges list
    M=length(E);
    N=max(max(E));
    A = sparse(E(:,1),E(:,2),ones(M,1),N,N,M);
    A = A+A';
end

