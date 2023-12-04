%this demo generates a network by block model, then compute DEA matrix of
%this network, compute eigenvalues of the DEA matrix and the overlap
path(path,'../../');
path(path,'../../subroutines/');
%%parameters
opt.N=1000; %system size
opt.c=3; %average degree
opt.epsilon=0.1; %cout/cin
opt.q=2; %number of groups
opt.numvec=6; %number of vectors you want
opt.seed=1;
opt.mode=5;

%%call deaspec
result=deaspec(opt);

%%output result
fprintf('First %d eigenvalues of DEA matrix are:\n',opt.numvec)
disp(result.D);
fprintf('Overlap to true configuration computed using sign of real eigenvectors are:\n');
disp(result.ovl);
