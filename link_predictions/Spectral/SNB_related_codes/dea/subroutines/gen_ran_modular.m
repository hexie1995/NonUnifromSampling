function [E,A,conf_true] = gen_ran_modular( N,c,q,p,seed )
% GEN_RAN_MODULAR generates random modular networks given N(system size), c(average degree)
% q (number of groups), p (epsilon=c_out/c_in) and seed(random number seed)

%     stream = RandStream.getGlobalStream; %for MATLAB 2013a
%    stream = RandStream.getDefaultStream; %for MATLAB 2008a
    ver=version('-release');
    if(strfind(ver,'201'))
        stream = RandStream.getGlobalStream; %for MATLAB 2012 and MATLAB 2013
    else
        stream = RandStream.getDefaultStream; %for MATLAB 2009 or earlier
    end
    assert(mod(N,q)==0);
    conf_true=[];
    for a=1:q
        conf_true = [conf_true ; ones(round(N/q),1)*a];
    end

    cin=q*c/(1+(q-1)*p);
    cout=cin*p;
    if(p>999)
        cin=0;
        cout=q*c/(q-1);
    end
    mmin=round(1/q*(N/q-1)/2*cin); % number of edges in same group
    mout=round(1/q*(N/q)*cout); % number of edges in different groups
    while true; % do until good
        reset(stream,seed);
        E=[];
        for a=1:q
            for b=a:q
                begina=round(N/q*(a-1)+1);
                enda=round(N/q*a);
                beginb=round(N/q*(b-1)+1);
                endb=round(N/q*b);
                if b==a % group aa
                    m=mmin;
                else
                    m=mout;
                end
                E1a=randi([begina enda],[round(1.2*m) 1]);
                E1b=randi([beginb endb],[round(1.2*m) 1]);
                E1=[E1a E1b];
                while (1)
                    E1=E1(find(E1(:,1)~=E1(:,2)),:);
                    E1T=E1';
                    [tmp,idx]=unique([min(E1T,[],1);max(E1T,[],1)]','rows');
                    E1=E1(idx,:);
                    if(length(E1)>=mmin)
                        break
                    else
                        E1a=randi([begina enda],[round(0.2*m) 1]);
                        E1b=randi([beginb endb],[round(0.2*m) 1]);
                        E1=[E1;E1a E1b];
                    end
                end
                
                %E1=E1(1:m,:);
                idx=randperm(length(E1));
                E1=E1(idx(1:m),:);
                E=[E;E1];
            end
        end
        Nx=max(max(E));
        if(Nx == N) 
            break;
        end
        seed=seed+1000;
    end
    M=length(E);
    A = sparse(E(:,1),E(:,2),ones(M,1),N,N,M);
    A = A+A';
    return
end

