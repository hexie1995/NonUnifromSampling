function [ V,D ] = get_spectrum( A,DEA,mode,numvec,numvec_eigs )
%GET_SPECTRUM compute spectrum of various matrices.
    if(numvec_eigs<numvec)
        numvec_eigs=numvec;
    end
    N=length(A);
    %A(A>1.1)=1;
    opts.maxit = 3000000;
    opts.disp=0;
    opts.isreal = 1;
    function y = dnRk(x); y = A*x-kit*(ki*x); end
    if(numvec>0)
        dprintf('computing eigs of ')
        if(mode==0)
            dprintf('DEA matrix\n');
            [V, D, flag]=eigs(DEA,numvec_eigs,'LM',opts);		
        elseif(mode==1)%modularity
            opts.issym = 1;
            ki=0;  kit=ki;
            ki=sum(A);
            tm=sum(ki);
            kit=ki'/tm;
            dprintf('Modularity matrix\n');
            [V,D,flag] = eigs(@dnRk,N,numvec_eigs,'LA',opts);
        elseif (mode==2)
            dprintf('Random walk matrix\n');
            kvec = (sum(A,1));
            Di = sparse(1:N,1:N,1./kvec,N,N,N);
            P = Di*A;
            [V,D,flag] = eigs(P,numvec_eigs,'LM',opts);
        elseif (mode==3)
            dprintf('Aij \n');
            opts.issym = 1;
            [V,D,flag] = eigs(A,numvec_eigs,'LA',opts);
            %D
        elseif(mode==4) % Lsym=D^-1/2(D-A)D^-1/2
            dprintf('Normalized Laplacian \n');
            D=A-A;
            D(speye(size(A))==1)=sum(A);
            L=D-A;
            Dm=D;
            Dm(Dm>0)=D(D>0).^(-0.5);
            Lsym=Dm*L*Dm;
            [V, D,flag] = eigs(Lsym,numvec,'SA',opts);
        elseif(mode==5) %B'
            dprintf('F \n');
            D=A-A;
            D(speye(size(A))==1)=sum(A);
            I=speye(size(A));
            F=[sparse(N,N) D-I;-I A];
            [V,D,flag] = eigs(F,numvec_eigs,'LM',opts);
            V=V(length(A)+1:2*length(A),:);
        else 
            printf('wrong mode\n');  return;
        end   
        if(flag==0)
            dprintf('all eigenvalues converged\n');
        else
            disp('not all eigenvalues converged\n');
        end
    else
        if(N>10000)
            fprintf('Are you sure you want to compute full spectrum of such a large matrix?');
            return;
        end
        dprintf('computing eig of ')
        if(mode==0)
            dprintf('DEA matrix\n');
            [V,D]=eig(full(DEA));
        elseif(mode==1)%modularity
            dprintf('Modularity matrix\n');
            disp('not ready yet!!!');
            return;
            %D = eig();
        elseif (mode==2)
            dprintf('Random walk matrix\n');
            kvec = (sum(A,1));
            Di = sparse(1:N,1:N,1./kvec,N,N,N);
            P = Di*A;
            [V, D] = eig(full(P));
        elseif (mode==3)
            dprintf('Aij \n');
            [V, D] = eig(full(A));
        elseif(mode==4) % Lsym=D^-1/2(D-A)D^-1/2
            D=A-A;
            D(speye(size(A))==1)=sum(A);
            L=D-A;
            Dm=D;
            Dm(Dm>0)=D(D>0).^(-0.5);
            Lsym=Dm*L*Dm;
            [V, D] = eig(full(Lsym));
        elseif(mode==5) % 
            dprintf('F \n');
            D=A-A;
            D(speye(size(A))==1)=sum(A);
            I=speye(size(A));
            F=[sparse(N,N) D-I;-I A];
            [V, D] = eig(full(F));
            V=V(length(A)+1:2*length(A),:);
        else 
            printf('wrong mode\n');  return;
        end   
        
    end
    
    %% sort eigenvalues
    D = sum(D);
    if(mode==4)
        [vtmp, ind]=sort(abs(D));
    else
        [vtmp, ind]=sort(abs(D),'descend');
    end
    D = D(ind);
    V = V(:,ind);
    if(numvec>0)
        D=D(1:numvec);
        V=V(:,1:numvec);
    end
end

