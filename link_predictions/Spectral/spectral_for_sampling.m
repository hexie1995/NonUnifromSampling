

function auc_score = spectral_for_sampling(path1, path2, net, samp)


epsilon = 1e-10;
rng(net)

addpath(genpath('/home/xhe/after_samp_process/link_predictions/Spectral/m1/'));
%path = 'C:\Users\hexie\OneDrive\Desktop\sampling\';

name = sprintf('net%d_',net);
name_str = sprintf('net%d_.txt', net);
test_name = sprintf('edge_set_test.txt');
train_name = sprintf('edge_set_train.txt');


edges_orig = dlmread([path1,name_str]);
edges_orig = edges_orig + 1;
N_orig = max(edges_orig,[],"all")+1;

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

train_name = sprintf('1%s_t_train.npy', samp)


str_EL_sampled = sprintf('%s_%s',name, train_name);
edges_train = dlmread([path2,str_EL_sampled]);
edges_train = edges_train + 1;
%N_train = length(unique(edges_train))
N_train =  N_orig
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

Ahat = V(:,1:k_NBM)*diag(D(1:k_NBM))*V(:,1:k_NBM).';

[nonedges_row,nonedges_col] = find((triu(Ao==0,1))==1);
nonedges_length = length(nonedges_row);

Nsamples = 10000;

t_sam_str = sprintf('%s_1_t_test_10000.npy',name);
f_sam_str = sprintf('%s_1_f_test_10000.npy',name);
f_samples = dlmread([path2, f_sam_str]);
t_samples = dlmread([path2, t_sam_str]);
f_samples = f_samples + 1;
t_samples = t_samples + 1;
results = [];
TP_aux = 0;
for ll=1:Nsamples
    edge_f = f_samples(ll,:);
    edge_t = t_samples(ll,:);

    acheck11 = Ahat(edge_t(1,1),edge_t(1,2));
    acheck12 = Ahat(edge_t(1,2),edge_t(1,1));
    if abs(acheck11 - acheck12)>epsilon
        disp("not equal")
        number_checker = number_checker + 1;
    end


    acheck21 = Ahat(edge_f(1,1),edge_f(1,2));
    acheck22 = Ahat(edge_f(1,2),edge_f(1,1));
    if abs(acheck21 - acheck22)>epsilon
        disp("not equal")
        number_checker = number_checker + 1;
    end


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