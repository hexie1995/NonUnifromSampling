function [ E ] = A2E( A )
%A2E compute edge list by adjacency matrix
    [I,J]=ind2sub(size(A),find(triu(A,1)>0));
    E=[I,J];
end

