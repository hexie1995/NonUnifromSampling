function [ V, D ] = get_approx_vec( A, DEA, DEAIM, mode, approx_r, conf_true )
%GET_APPROX_VEC compute approximate 2nd eigenvector of DEA matrix.
disp('computing approximate eigenvector');
assert(approx_r>0);
assert(max(conf_true)==2);
assert(mode==0 || mode==3);
sigma=conf_true;
sigma(conf_true==1)=-1;
sigma(conf_true==2)=1;
V=[];
D=[];
if(mode==0)
    %first step
    opts.isreal=1;
    [Vx, Dx, flag]=eigs(DEA,2,'LM',opts);
    %initial matrix
    v1=Vx(:,1);
    V0=DEAIM*sigma;
    
    V0=V0-dot(V0,v1)*v1; %project away the first eigenvector.
    V0=V0./norm(V0);
    
    %D=[D 1];
    V=[V V0]; 
    
    %more steps
    for i=1:approx_r-1
        V1=DEA*V0;
        D1=mean(V1(V0>1.0e-5)./V0(V0>1.0e-5));
        D=[D D1];
        V1=V1./norm(V1);
        V=[V V1];
        V0=V1;
        V0=V0-dot(V0,v1)*v1;%project away the first eigenvector.
        V0=V0./norm(V0);
    end
else
    V0=sigma;
    for i=1:approx_r
        V1=A*V0;
        D1=mean(V1(V0>1.0e-5)./V0(V0>1.0e-5));
        D=[D D1];
        V1=V1./norm(V1);
        V=[V V1];
        V0=V1;
    end
end

