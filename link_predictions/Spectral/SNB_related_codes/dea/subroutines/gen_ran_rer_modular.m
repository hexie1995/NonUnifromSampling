function [E,A,conf_true] = gen_ran_rer_modular( N,c,q,p,seed )
% GEN_RAN_RER_MODULAR generates random modular random-regular networks given N(system size), c(degree)
% q (number of groups), p (epsilon=c_out/c_in) and seed(random number seed)

     stream = RandStream.getGlobalStream; %for MATLAB 2012b
    disp('regular random networks');
    %stream = RandStream.getDefaultStream; %for MATLAB 2008a
    reset(stream,seed);
    K=round(c);
    assert(mod(N,q)==0);
    conf_true=[];
    for a=1:q
        conf_true = [conf_true ; ones(round(N/q),1)*a];
    end
    slots_orig=linspace(1,N*K,N*K);
    slots_used=[];

    %cin=q*c/(1+(q-1)*p);
    cin= 0.5*K*N/(( (N/q-1)/2) + (q-1)*N*p/q/2 );
    cout=cin*p;
    if(p>999)
        cin=0;
        cout=q*c/(q-1);
    end
    mmin=round(1/q*(N/q-1)/2*cin); % number of edges in same group
    %mout=round(1/q*(N/q)*cout); % number of edges in different groups
    mout=round( (K*N/2-mmin*q)*2/(q-1)/q);
    %[mmin mout]
    %[(mmin*q+mout*(q-1)*q/2)*2 K*N]
    assert(abs((mmin*q+mout*(q-1)*q/2)*2-K*N)<0.5);
    
    slots_ava=slots_orig; %avilable configurations used in matching.
    E=[];
    m=mmin;
    for a=1:q
        begina=round((a-1)*N/q)*K+1;
        enda=round(a*N/q)*K;
        slotsa=slots_ava( logical( (slots_ava >= begina ) .* (slots_ava <= enda)) );
        slotsa=slotsa(randperm(length(slotsa)));
        E1a=slotsa(1:2*m)';
        E1=[E1a(1:m) E1a(m+1:2*m)];
        E=[E; E1];
        slots_used=[slots_used E1a'];
        slots_ava=setxor(slots_ava,E1a');
    end
    %[length(slots_orig) length(slots_ava) length(slots_used)
    %[length(unique(slots_ava)) length(unique(slots_used)) ]
    
    m=mout;
    for a=1:q
        for b=(a+1):q
            begina=round((a-1)*N/q)*K+1;
            enda=round(a*N/q)*K;
            slotsa=slots_ava( logical( (slots_ava >= begina ) .* (slots_ava <= enda)) );
            slotsa=slotsa(randperm(length(slotsa)));
            E1a=slotsa(1:m)';
            
            beginb=round((b-1)*N/q)*K+1;
            endb=round(b*N/q)*K;
            slotsb=slots_ava( logical( (slots_ava >= beginb ) .* (slots_ava <= endb)) ); 
            slotsb=slotsb(randperm(length(slotsb)));
            E1b=slotsb(1:m)';
            E1=[E1a E1b];

            E=[E; E1];
            slots_used=[slots_used E1a' E1b'];
            slots_ava=setxor(slots_ava,E1a');
            slots_ava=setxor(slots_ava,E1b');
        end
    end
                 
    %[length(slots_orig) length(slots_ava) length(slots_used) length(unique(slots_ava)) length(unique(slots_used)) ]
    E=ceil(E./K);
    M=length(E);
    A = sparse(E(:,1),E(:,2),ones(M,1),N,N,M);
    A = A+A';
%     %% deal with self-connections
%     disp('removing self-edges...');

%     idxself=find(E(:,1)==E(:,2));
%     while(length(idxself)>0)
%         for i=1:length(idxself)
%             idx=idxself(i);
%             swapE;
%         end
%         idxself=find(E(:,1)==E(:,2));
%     end
%     
%     %% deal with multi-connection
%     disp('removing multi-edges...');
%     E1T=E';
%     %A = sparse(E(:,1),E(:,2),ones(M,1),N,N,M);
%     %A = A+A';
%     idxall=linspace(1,length(E),length(E));
%     [tmp,idxu]=unique([min(E1T,[],1);max(E1T,[],1)]','rows');
%     idxdup=sort(setxor(idxall,idxu));
%     while(length(idxdup)>0)
%         for i=1:length(idxdup)
%             idx=idxdup(i);
%             swapE;
%         end
%         E1T=E';
%         [tmp,idxu]=unique([min(E1T,[],1);max(E1T,[],1)]','rows');
%         idxdup=sort(setxor(idxall,idxu));
%     end
%     
%     %% Mente-carlo swapping
%     disp('Monte-Carlo swaping');
%     for i=1:10*length(A);
%         idx=randi([1 length(E)],1,1);
%         swapE;
%     end


%    di=sum(A);
%    A
%    E
    %imshow(full(A))
end

