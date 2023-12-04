function [D,conf_infer,Vsort]=compute_conf_infer(Q,D,V,numvec_,do_clustering,mode,cbegin,cend,only_real_,smallest_,conf_true, Am)
    %COMPUTE_CONF_INFER compute a configuration from eigenvectors
    only_real=1;
    numvec=abs(numvec_);
	smallest=0;
    if(nargin>=10)
		only_real=only_real_;
    end
	if(nargin>=11)
		smallest=smallest_;
    end
    na=ones(1,Q)./Q;
    %for i=1:Q
    %    na(i)=sum(conf_true == i)/numel(conf_true);
    %end
    if(only_real==1)
        fprintf('only consider real eigenvalues\n');
        idx=abs(imag(D))<0.0001;
        D=real(D(idx));
        Vsort=V(:,idx);
        numhave=sum(idx);
    else
        fprintf('consider all eigenvalues\n');
        Vsort=real(V);
%         Vsort=V;
%         ridx=abs(imag(Vsort))<0.01;
%         iidx=abs(imag(Vsort))>=0.01;
%         Vsort(ridx)=real(Vsort(ridx));
%         Vsort(iidx)=imag(Vsort(iidx));
        numhave=numel(V(1,:));
    end
    
    
    if(numhave<cbegin)
        fprintf('number of real eigenvalue %d is smaller than %d, so I calculate using vectors from %d to %d\n',1,numhave);
        cbegin=1;
        cend=numhave;
    elseif(numhave < cend)
        fprintf('number of real eigenvalue %d is smaller than %d, so I calculate using vectors from %d to %d\n',numhave,cend);
        cend=numhave;
    end
    
    if(mode==4)%L_sym, use second smallest eigenvector
        [mytmp,idx]=sort(abs(D));
        D=D(idx);
        Vsort=Vsort(:,idx);
    end
	if(mode==0) % DEA matrix
		Sv=Am*Vsort;
	else
		Sv=Vsort;
	end
    Vsort=Sv;
    %assert(numel(Sv(:,1)) == numel(conf_true));
    conf_infer=[];
    if(do_clustering == 0)
		fprintf('threshold 0 using vectors from %d to %d\n',cbegin,cend);
        %ovl=[];
        for num=cbegin:cend
            sigma=ones(numel(Sv(:,num)),1);
            sigma(abs(Sv(:,num))<eps)=0;
            sigma(Sv(:,num)>=eps)=2;
            conf_infer=[conf_infer, sigma];
        end
    elseif(do_clustering == 1)
		fprintf('using na using vectors from %d to %d\n',cbegin,cend);
        fprintf('na:')
        disp(na)
        %ovl=[];
        for num=cbegin:cend
            sigma=ones(numel(Sv(:,numhave)),1);
            [mytmp,oidx]=sort(Sv(:,num)); 
            N=numel(oidx);
            for i=1:Q
                if(i==1)
                    sigma(oidx(1:round(N*na(i))))=1;
                else
                    sigma(oidx( round( N*na(i-1))+1:round(N*( na(i-1)+na(i) ) ) ))=i;
                end
            end
            conf_infer=[conf_infer, sigma];
        end
    elseif(do_clustering==2 || do_clustering==3 || do_clustering==4)
        fprintf('k-means from %d to %d\n',cbegin,cend);
        Sv=Sv(:,cbegin:cend);
        if(do_clustering==4) 
            fprintf('change basis by PCA\n');
            [tmp,Sv]=pca(Sv);
        elseif(do_clustering==3)
            fprintf('each row of vectors are normalized\n');
            for x=1:length(Sv);
                if(norm(Sv(x,:))>0.00001)
                    Sv(x,:)=Sv(x,:)/norm(Sv(x,:));
                end
            end
        else
            fprintf('each colume of vectors are normalized\n');
            %for x=1:length(Sv(1,:));
            %    Sv(:,x)=Sv(:,x)/norm(Sv(:,x));
            %end
        end
        %sigma=kmeans(Sv(:,cbegin:cend),Q);
        sigma=kmeans(real(Sv),Q);
        conf_infer=[conf_infer, sigma];
        inn=[];
        for i=1:length(Sv(1,:))
            for j=i+1:length(Sv(1,:))
                inn=[inn dot(Sv(:,i),Sv(:,j))];
            end
        end
        %inn
    end
    
    D=D(cbegin:cend);
    Vsort=Vsort(:,cbegin:cend);
end