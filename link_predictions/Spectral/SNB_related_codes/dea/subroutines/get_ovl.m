function [ovls]=get_ovl(conf_true,conf_infer,ovl_norm_)
    % GET_OVL compute overlap between conf_true and conf_infer. If ovl_norm
    % is 1, the overlap value is normalized.
    ovl_norm=0;
    if(nargin >=3)
        ovl_norm=ovl_norm_;
    end
    Q=max(conf_true);
    na=ones(1,Q)./Q;
    %for i=1:Q
    %    na(i)=sum(conf_true == i)/numel(conf_true);
    %end
    maxna=max(na);
    if(Q>8)
        %[conf_true,conf_infer]
        %% normalized mutual information
        N=length(conf_true);
        nb=ones(1,Q)./Q
        for i=1:Q
            nb(i)=sum(conf_true == i);
        end
        Q1=zeros(Q,N);
        Q2=zeros(Q,N);
        for i=1:N
            Q1(conf_true(i),i)=1;
            Q2(conf_infer(i),i)=1;
        end
        nmi=mynmi(Q1,Q2)
                sigma=conf_infer(:,1);
        maxovl=-1;
        mysigma=sigma;
        for i=1:200000
            %swap a configuration
            a=randi(Q,1,1);
            b=randi(Q,1,1);
            while (a==b)
                b=randi(Q,1,1);
            end
            idxa=(sigma==a);
            idxb=(sigma==b);
            sigma(idxa)=b;
            sigma(idxb)=a;
            same=0;
            zcount=0;
            for j=1:numel(conf_true)
                if(sigma(j)<eps)
                    %same = same + 1.0/Q;
                    same = same + maxna;
                    zcount = zcount+1;
                else
                    if(conf_true(j)==sigma(j))
                        same = same+1.0;
                    end
                end
            end  
            ovl=same/numel(conf_true);
            if (ovl>maxovl)
                maxovl=ovl;
                %mysigma=sigma;
            else
                sigma(idxa)=a;
                sigma(idxb)=b;
            end
            maxovl;
        end
        %[conf_true mysigma]
        fprintf('maxovl=%g normalized=%g\n',maxovl,(maxovl-1/12)/(1-1/12));
        ovls=maxovl;
        %ovls=nmi;
        return;
    end
    pes=perms(linspace(1,Q,Q));
    ovls=[];
    for i=1:numel(conf_infer(1,:))
        sigma=conf_infer(:,i);
        ovl=0;
        for pe=1:length(pes)
            same=0;
            zcount=0;
            for j=1:numel(conf_true)
                if(sigma(j)<eps)
                    %same = same + 1.0/Q;
                    same = same + maxna;
                    zcount = zcount+1;
                else
                    if(pes(pe,conf_true(j))==sigma(j))
                        same = same+1.0;
                    end
                end
            end
            if(same/numel(conf_true)>ovl)
                ovl=same/numel(conf_true);
            end
        end
        if(ovl_norm)
            %disp('doing normalization')
            ovl=(ovl-maxna)/(1-maxna);
        end
        %[ovlold ovl]
        ovls=[ovls ovl];
    end
end


