function [ cut,Rcut,Ncut ] = get_cuts( E,confs )
    N=numel(confs(:,1));
    A=E2A(E);
    di=sum(A);
    Q=max(max(confs));
    %size(E)
    cut=[]; Rcut=[]; Ncut=[];
    for j=1:numel(confs(1,:))
        conf=confs(:,j);
        for i=1:N;
            if(conf(i)==0)
                conf(i)=randi(Q,1,1);
            end
        end
        %size(di)
        %size(conf)
        vol1=sum(di(conf==1));
        vol2=sum(di(conf==2));
        n1=sum(conf==1);
        n2=sum(conf==2);
        S1=conf(E(:,1));
        S2=conf(E(:,2));
        %S=(S1-S2).^2;
        %cut0=sum(S);
        cut0 = sum(S1 ~= S2);
        cut=[cut cut0];
        Ncut=[Ncut cut0*(1/vol1+1/vol2)];
        Rcut=[Rcut cut0*(1/n1+1/n2)];
    end
end

