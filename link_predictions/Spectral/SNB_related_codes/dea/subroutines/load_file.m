function [ E, A, conf_true ] = load_file( ftype, fname,lcc )
%LOAD_FILE Summary of this function goes here
%   Detailed explanation goes here
    if(strcmp(ftype,'nl'))
        [ E, A, conf_true ] = load_nl(fname);
    elseif (strcmp(ftype,'gml'))
        [ E, conf_true] = read_gml(fname);
        A=E2A(E);
    end
    if(lcc)
        %path(path,'./gaimc');
        disp('larget connected component');
        fprintf('running on largest connected component\n');
        [A,pt]=largest_component(A);
        Nt=length(pt);%original size
        N=length(A);%size of largest connected component
        fprintf('number of nodes: %d %d\n',Nt,N);
        E=A2E(A);
        M=length(E);
        conf_true=conf_true(pt);
    end
end

