

function auc_score = spectral_for_sampling(path, net, samp)


epsilon = 1e-10;
rng(net)

addpath(genpath('/home/xhe/SPECTRAL_5_RUN/spectral_1/'));
%path = 'C:\Users\hexie\OneDrive\Desktop\sampling\';

name = sprintf('net%d_',net);
name_str = sprintf('net%d_.txt', net);
test_name = sprintf('edge_set_test.txt');
train_name = sprintf('edge_set_train.txt');


edges_orig = dlmread([path,name_str]);
edges_orig = edges_orig + 1;
N_orig = length(unique(edges_orig));
Aorig = sparse(zeros(N_orig,N_orig));
M_orig = size(edges_orig,1);
for mm = 1:M_orig
    Aorig(edges_orig(mm,1),edges_orig(mm,2)) = 1;
    Aorig(edges_orig(mm,2),edges_orig(mm,1)) = 1;
end

N = N_orig;
A = Aorig;


results_foldername = sprintf('./LP_NBMvAST_Results/');
if exist(results_foldername,'dir') ~= 7
    mkdir(results_foldername)
end

AUC=0;


str_EL_sampled = sprintf('%s%s',name, train_name);
edges_train = dlmread([path,str_EL_sampled]);
edges_train = edges_train + 1;
N_train = length(unique(edges_train))
A_train = sparse(zeros(N_train,N_train));

M_train = size(edges_train,1);

for mm = 1:M_train
    A_train(edges_train(mm,1),edges_train(mm,2)) = 1;
    A_train(edges_train(mm,2),edges_train(mm,1)) = 1;
end

Ao = A_train;
N_e = N_train;

eig_num = min(round(2*sqrt(N_e)), N);
opts_tol_opt=eps;

tr=1;
opts.tol=opts_tol_opt;


while tr==1
    try
        opts.tol=opts_tol_opt;
        [k_NBM,r] = via_nonbacktracking_AG(Ao,eig_num,opts);
        tr=0;
    catch
        disp('error')
        opts_tol_opt=opts_tol_opt*10000;
    end
end
%
if k_NBM == 0
    k_NBM = 1;
end
%label_inferred_aux = dlmread(sprintf('./label_S_NB/label_SNB_f_%d_n_%d_%d.txt',frac_id,nsim_id,matrix_id));
%k_NBM = length(unique(label_inferred_aux));

if(k_NBM>2)
    do_clustering=3; % if q>2, use k-means clustering
else
    do_clustering=0; % if q==2, use sign of second largest real eigenvector
end
%
n = size(Ao,1); d = sum(Ao,1); tolerance = 10^(-5);
%
B = [zeros(n), diag(d-1); -eye(n), Ao]; % non-backtracking matrix
%
%%%%%%%%%%%%
DEA=0; DEAM=0; DEAIM=0;
mode = 5;
%
num_vec = -1; numvec_eigs = k_NBM;
[ V_,D_,V_clustering ] = get_spectrum_AG( Ao, DEA, mode, num_vec, numvec_eigs);
%
vl=0;
ovl_real = 1;
cbegin = 1;
cend = k_NBM;
q=k_NBM;
conf_true = zeros(N,1);
numvec = -1;
[Dsort_tmp,conf_infer,Vsort_tmp]=compute_conf_infer(q,D_,V_clustering,numvec,do_clustering,mode,cbegin,cend,ovl_real,0,conf_true,DEAM);
label_foldername = sprintf('./label_S_NB/');
if exist(label_foldername,'dir') ~= 7
    mkdir(label_foldername)
end

dlmwrite(sprintf('./label_S_NB/label_SNB_f_%s_%s.txt', name, samp),conf_infer(:,end));
%%%%%%%%%%%%

opts_tol_opt=eps;

tr=1;
opts.tol=opts_tol_opt;


while tr==1
    try
        opts.tol=opts_tol_opt;
        [V,D] = eigs(full(Ao),k_NBM,'lm',opts);
        tr=0;
    catch
        disp('error')
        opts_tol_opt=opts_tol_opt*10000;
    end
end
D = sum(D);
[vtmp, ind]=sort(abs(D),'descend');
D = D(ind);
V = V(:,ind);
% [V,D] = eigs(Ao,k_NBM,'lr'); eigval = diag(D);
Ahat = V(:,1:k_NBM)*diag(D(1:k_NBM))*V(:,1:k_NBM).';
%         Bphat = V(:,1:k_NBM)*diag(D(1:k_NBM))*V(:,1:k_NBM).';
[nonedges_row,nonedges_col] = find((triu(Ao==0,1))==1);
nonedges_length = length(nonedges_row);

%         A_diff = Aorig - Ao;
%         [t_edges_row,t_edges_col] = find(triu(A_diff,1)==1);
%         [nt_edges_row,nt_edges_col] = find((triu(Aorig==0,1))==1);
Nsamples = 10000;

t_sam_str = sprintf('%s_2%s_t_train_10000.npy',name, samp);
f_sam_str = sprintf('%s_2%s_f_train_10000.npy',name, samp);
f_samples = dlmread([path, f_sam_str]);
t_samples = dlmread([path, t_sam_str]);
%f_samples = dlmread(sprintf('../../all_netws/edge_tf/edge_f_frac_%d_nsim_%d_%d.txt',frac_id,nsim_id,matrix_id));
%t_samples = dlmread(sprintf('../../all_netws/edge_tf/edge_t_frac_%d_nsim_%d_%d.txt',frac_id,nsim_id,matrix_id));
f_samples = f_samples + 1;
t_samples = t_samples + 1;
results = [];
TP_aux = 0;
for ll=1:Nsamples
    %             edge_t_idx = randi(length(t_edges_row));
    %             edge_f_idx = randi(length(nt_edges_row));
    edge_f = f_samples(ll,:);
    edge_t = t_samples(ll,:);

    acheck11 = Ahat(edge_t(1,1),edge_t(1,2));
    acheck12 = Ahat(edge_t(1,2),edge_t(1,1));
    if abs(acheck11 - acheck12)>epsilon
        disp("not equal")
        number_checker = number_checker + 1;
    end

    %A_aux_t = full(Ao);
    %A_aux_t(edge_t(1,1),edge_t(1,2)) = 1;
    %A_aux_t(edge_t(1,2),edge_t(1,1)) = 1;
    %dQ_t = 1/norm(Ahat - A_aux_t,'fro') + rand(1)/(100000*N);


    acheck21 = Ahat(edge_f(1,1),edge_f(1,2));
    acheck22 = Ahat(edge_f(1,2),edge_f(1,1));
    if abs(acheck21 - acheck22)>epsilon
        disp("not equal")
        number_checker = number_checker + 1;
    end

    %A_aux_f = full(Ao);
    %A_aux_f(edge_f(1,1),edge_f(1,2)) = 1;
    %A_aux_f(edge_f(1,2),edge_f(1,1)) = 1;

    %dQ_f = 1/norm(Ahat - A_aux_f,'fro') + rand(1)/(100000*N);

    if (acheck11 > acheck21) && (acheck12 > acheck22) && abs(acheck11 - acheck21)>epsilon
        TP_aux = TP_aux + 1;
        results = [results;edge_t edge_f acheck11 acheck21 1];
    elseif (abs(acheck11 - acheck21)<=epsilon)
        if rand(1)<0.5
            TP_aux = TP_aux + 1;
            results = [results;edge_t edge_f acheck11 acheck21 1];
        else
            results = [results;edge_t edge_f acheck11 acheck21 0];
        end
    else
        results = [results;edge_t edge_f acheck11 acheck21 0];
    end


    %if dQ_t > dQ_f
    %    TP_aux = TP_aux + 1;
    %    results = [results;edge_t edge_f dQ_t dQ_f 1];
    %elseif dQ_t == dQ_f
    %    if randi(2)==1
    %        TP_aux = TP_aux + 1;
    %        results = [results;edge_t edge_f dQ_t dQ_f 1];
    %    else
    %        results = [results;edge_t edge_f dQ_t dQ_f 0];
    %    end
    %else
    %    results = [results;edge_t edge_f dQ_t dQ_f 0];
    %end
end
AUC= TP_aux/Nsamples;
if net == 0
    results_foldername = sprintf('./LP_NBMvAST_Results/');
    if exist(results_foldername,'dir') ~= 7
        mkdir(results_foldername)
    end
    fid = fopen(sprintf('%sNBMvAvs_LP_%s_%s.txt',results_foldername, name, samp),'w');
    fprintf(fid,'%f\n',AUC);
    fclose(fid);
else
    results_foldername = sprintf('./LP_NBMvAST_Results/');
    fid = fopen(sprintf('%sNBMvAvs_LP_%s_%s.txt',results_foldername, name, samp),'a');
    fprintf(fid,'%f\n',AUC);
    fclose(fid);
end
dlmwrite(sprintf("%sresultsNBM_%s_%s.txt",results_foldername,name, samp),results);

auc_score = AUC
end