
% % ------------------------------

% % ------------------------------
function auc_score = modularity_for_sampling(path1, path2, net, samp)

rng(net)
addpath(genpath('/home/xhe/after_samp_process/link_predictions/Modularity/m1/Modularity_related_codes/'));
addpath(genpath('/home/xhe/after_samp_process/link_predictions/Modularity/m1/'));
%path = 'C:\Users\hexie\OneDrive\Desktop\sampling\';


name = sprintf('net%d_',net);
name_str = sprintf('net%d_.txt', net);

%str = sprintf('edge_list_LC_%d.txt',gml_id);
edges_orig = dlmread([path1,name_str]);
edges_orig = edges_orig+1;
%edges_orig = [1 2;1 3;2 3;3 4;4 5;4 6;5 6];




% FP=cell(length(frac_id_set),length(count_set));
% TP=cell(length(frac_id_set),length(count_set));
%%
N_orig = max(edges_orig,[],"all")+1;
A_orig = sparse(zeros(N_orig,N_orig));
M_orig = size(edges_orig,1);
for mm = 1:M_orig
    A_orig(edges_orig(mm,1),edges_orig(mm,2)) = 1;
    A_orig(edges_orig(mm,2),edges_orig(mm,1)) = 1;
end
results_foldername = sprintf('./LP_Mod_ResultsSTST/');
if exist(results_foldername,'dir') ~= 7
    mkdir(results_foldername)
end


AUC=0 ;
labels_foldername_Q = sprintf('./label_Q/');
if exist(labels_foldername_Q,'dir') ~= 7
    mkdir(labels_foldername_Q)
end

train_name = sprintf('1%s_t_train.npy', samp)


str_EL_sampled = sprintf('%s_%s',name, train_name);
fid = fopen([path2,str_EL_sampled]);
tline_aux = fgets(fid);

edges_train = dlmread([path2,str_EL_sampled]);
N_train = N_orig;


edges = [];
while ischar(tline_aux)
    if strfind(tline_aux,'# # of nodes:')
        num_orig_nodes = tline_aux(strfind(tline_aux,':')+2:end);
    elseif strfind(tline_aux,'# fraction of edges revealed:')
        frac_edges_revealed = tline_aux(strfind(tline_aux,':')+2:end);
    elseif strfind(tline_aux,'# number of original edges:')
        num_orig_edges = tline_aux(strfind(tline_aux,':')+2:end);
    elseif strfind(tline_aux,'# number of edges revealed:')
        num_edges_revealed = tline_aux(strfind(tline_aux,':')+2:end);
    else
        edges = [edges;str2num(tline_aux)];
    end
    tline_aux = fgets(fid);
end
fclose(fid);
edges = edges + 1;


N = N_train;
Ao = sparse(zeros(N,N));
M = size(edges,1);
for mm = 1:M
    Ao(edges(mm,1),edges(mm,2)) = 1;
    Ao(edges(mm,2),edges(mm,1)) = 1;
end
%Ao=[0 0 1 0 0 0;0 0 1 0 0 0;1 1 0 1 0 0;0 0 1 0 1 1;0 0 0 1 0 0;0 0 0 1 0 0];
%Ao=[0 1 1 0 0 0;1 0 1 0 0 0;1 1 0 0 0 0;0 0 0 0 1 1;0 0 0 1 0 1;0 0 0 1 1 0];
num_nodes = size(Ao,1);
max_itr = 100;
min_itr = 10;
Num_Sim = 5;
Q = zeros(Num_Sim,1);
Q_gamma1 = zeros(Num_Sim,1);
S = zeros(num_nodes,Num_Sim);
S_gamma1 = zeros(num_nodes,Num_Sim);
q_inferred = zeros(Num_Sim,1);
q_inferred_gamma1 = zeros(Num_Sim,1);
gamma_f = zeros(Num_Sim,1);
convergence_checker = 0;

for num_sim = 1:Num_Sim
    gamma = 1;
    k = full(sum(Ao));
    twom = sum(k);
    B = full(Ao - gamma*(k.')*k/twom);
    [S(:,num_sim),Q(num_sim)] = genlouvain(B);
    Q(num_sim) = Q(num_sim)/twom;

    q_inferred(num_sim) = length(unique(S(:,num_sim)));

    gamma_f(num_sim) = gamma;
end
[Q_f,id_f] = max(Q);
S_f = S(:,id_f);
%S_f_mat = cell2mat(S_f)

dlmwrite(sprintf('./%s/label_Q_%s_%s.txt',labels_foldername_Q, name, samp),S_f);
%         A_diff = A_orig - Ao;
%         [t_edges_row,t_edges_col] = find(triu(A_diff,1)==1);
%
%         [nt_edges_row,nt_edges_col] = find((triu(A_orig==0,1))==1);

%         [nonedges_row,nonedges_col] = find((triu(Ao==0,1))==1);
%         nonedges_length = length(nonedges_row);
%%
twoL = twom;
unique_types = unique(S_f);
Qa = zeros(size(unique_types));
Qa_2 = zeros(size(unique_types));
twoLa = zeros(size(unique_types));
twoLab = zeros(length(unique_types),length(unique_types));
Da = zeros(size(unique_types));
for tt=1:length(unique_types)
    Ia = (S_f==tt);
    twoLa(tt) = (Ia.')*Ao*Ia;
    Da(tt) = sum(sum((Ia.')*Ao));
    Qa(tt) =  twoLa(tt)/twoL-(Da(tt)/twoL)^2;
    Qa_2(tt) =  twoLa(tt)/(twoL+2)-(Da(tt)/(twoL+2))^2;
end
%         Qab=zeros(length(unique_types));
Qab_a=zeros(length(unique_types));
for tt1=1:length(unique_types)
    Ia = (S_f==tt1);
    for tt2=tt1+1:length(unique_types)
        Ib = (S_f==tt2);
        twoLab(tt1,tt2) = 2*(Ia.')*Ao*Ib;
        Qab_a(tt1,tt2) = twoLa(tt1)/(twoL+2) - ((Da(tt1) + 1)/(twoL+2))^2 + ...
            twoLa(tt2)/(twoL+2) - ((Da(tt2) + 1)/(twoL+2))^2;
        Qab_a(tt2,tt1) = Qab_a(tt1,tt2);
        %                 Qab(tt1,tt2) = (twoLa(tt1)+twoLa(tt2)+twoLab(tt1,tt2)+2)/(twoL+2) - ((Da(tt1)+Da(tt2)+2)/(twoL+2))^2;
        %                 Qab(tt2,tt1) = Qab(tt1,tt2);
    end
end
twoLia = zeros(N,length(unique_types));
Di = sum(Ao,2);
for nn=1:N
    for tt=1:length(unique_types)
        Ia = (S_f==tt);
        twoLia(nn,tt) = 2*Ao(nn,:)*Ia;
    end
end
%         true_labels = zeros(nonedges_length,1);
%         dQ = zeros(nonedges_length,1);
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
    %             if sum(prod([nonedges_row(ll)==edges_orig(:,1) nonedges_col(ll)==edges_orig(:,2)],2)==1)==1 ||...
    %                     sum(prod([nonedges_col(ll)==edges_orig(:,1) nonedges_row(ll)==edges_orig(:,2)],2)==1)==1
    %                 true_labels(ll) = 1;
    %             end
    %             edge_t_idx = randi(length(t_edges_row));
    %             edge_f_idx = randi(length(nt_edges_row));
    edge_f = f_samples(ll,:);
    edge_t = t_samples(ll,:);

    if (S_f(edge_t(1,1))==S_f(edge_t(1,2)))
        %                 Q1 = Qa(S_f(nonedges_row(ll)));
        Q1 = sum(Qa);
        %                 Q2 = (twoLa(S_f(nonedges_row(ll)))+2)/(twoL+2)-((Da(S_f(nonedges_row(ll)))+2)/(twoL+2))^2;
        Q2 = sum(Qa_2([1:S_f(edge_t(1,1))-1 S_f(edge_t(1,1))+1:end],1)) + (twoLa(S_f(edge_t(1,1)))+2)/(twoL+2)-((Da(S_f(edge_t(1,1)))+2)/(twoL+2))^2;
    else
        %                 Q1 = Qa(S_f(nonedges_row(ll))) + Qa(S_f(nonedges_col(ll)));
        Q1 = sum(Qa);
        %                 Q2 = Qab(S_f(nonedges_row(ll)),S_f(nonedges_col(ll)));
        if (S_f(edge_t(1,1)) < S_f(edge_t(1,2)))
            type_min = S_f(edge_t(1,1));
            type_max = S_f(edge_t(1,2));
        else
            type_max = S_f(edge_t(1,1));
            type_min = S_f(edge_t(1,2));
        end
        %                 Q2_ijmerge = sum(Qa_2([1:type_min-1 type_min+1:type_max-1 type_max+1:end],1)) + Qab(S_f(edge_t(1,1)),S_f(edge_t(1,2)));
        %                 Q2_imergej = sum(Qa_2([1:type_min-1 type_min+1:type_max-1 type_max+1:end],1)) + ...
        %                     (twoLa(S_f(edge_t(1,1)))-twoLia(edge_t(1,1),S_f(edge_t(1,1))))/(twoL+2) - ((Da(S_f(edge_t(1,1)))-Di(edge_t(1,1),1))/(twoL+2))^2 + ...
        %                     (twoLa(S_f(edge_t(1,2)))+twoLia(edge_t(1,1),S_f(edge_t(1,2)))+2)/(twoL+2) - ((Da(S_f(edge_t(1,2)))+Di(edge_t(1,1),1)+2)/(twoL+2))^2;
        %                 Q2_jmergei = sum(Qa_2([1:type_min-1 type_min+1:type_max-1 type_max+1:end],1)) + ...
        %                     (twoLa(S_f(edge_t(1,2)))-twoLia(edge_t(1,2),S_f(edge_t(1,2))))/(twoL+2) - ((Da(S_f(edge_t(1,2)))-Di(edge_t(1,2),1))/(twoL+2))^2 + ...
        %                     (twoLa(S_f(edge_t(1,1)))+twoLia(edge_t(1,2),S_f(edge_t(1,1)))+2)/(twoL+2) - ((Da(S_f(edge_t(1,1)))+Di(edge_t(1,2),1)+2)/(twoL+2))^2;
        %                 Q2 = max([Q2_ijmerge Q2_imergej Q2_jmergei]);
        Q2 = sum(Qa_2([1:type_min-1 type_min+1:type_max-1 type_max+1:end],1)) + Qab_a(S_f(edge_t(1,1)),S_f(edge_t(1,2)));
    end
    dQ_t = (Q2 - Q1)*1.0 + rand(1)/(1000*N);

    if (S_f(edge_f(1,1))==S_f(edge_f(1,2)))
        %                 Q1 = Qa(S_f(nonedges_row(ll)));
        Q1 = sum(Qa);
        %                 Q2 = (twoLa(S_f(nonedges_row(ll)))+2)/(twoL+2)-((Da(S_f(nonedges_row(ll)))+2)/(twoL+2))^2;
        Q2 = sum(Qa_2([1:S_f(edge_f(1,1))-1 S_f(edge_f(1,1))+1:end],1)) + (twoLa(S_f(edge_f(1,1)))+2)/(twoL+2)-((Da(S_f(edge_f(1,1)))+2)/(twoL+2))^2;
    else
        %                 Q1 = Qa(S_f(nonedges_row(ll))) + Qa(S_f(nonedges_col(ll)));
        Q1 = sum(Qa);
        %                 Q2 = Qab(S_f(nonedges_row(ll)),S_f(nonedges_col(ll)));
        if (S_f(edge_f(1,1)) < S_f(edge_f(1,2)))
            type_min = S_f(edge_f(1,1));
            type_max = S_f(edge_f(1,2));
        else
            type_max = S_f(edge_f(1,1));
            type_min = S_f(edge_f(1,2));
        end
        %                 Q2_ijmerge = sum(Qa_2([1:type_min-1 type_min+1:type_max-1 type_max+1:end],1)) + Qab(S_f(edge_f(1,1)),S_f(edge_f(1,2)));
        %                 Q2_imergej = sum(Qa_2([1:type_min-1 type_min+1:type_max-1 type_max+1:end],1)) + ...
        %                     (twoLa(S_f(edge_f(1,1)))-twoLia(edge_f(1,1),S_f(edge_f(1,1))))/(twoL+2) - ((Da(S_f(edge_f(1,1)))-Di(edge_f(1,1),1))/(twoL+2))^2 + ...
        %                     (twoLa(S_f(edge_f(1,2)))+twoLia(edge_f(1,1),S_f(edge_f(1,2)))+2)/(twoL+2) - ((Da(S_f(edge_f(1,2)))+Di(edge_f(1,1),1)+2)/(twoL+2))^2;
        %                 Q2_jmergei = sum(Qa_2([1:type_min-1 type_min+1:type_max-1 type_max+1:end],1)) + ...
        %                     (twoLa(S_f(edge_f(1,2)))-twoLia(edge_f(1,2),S_f(edge_f(1,2))))/(twoL+2) - ((Da(S_f(edge_f(1,2)))-Di(edge_f(1,2),1))/(twoL+2))^2 + ...
        %                     (twoLa(S_f(edge_f(1,1)))+twoLia(edge_f(1,2),S_f(edge_f(1,1)))+2)/(twoL+2) - ((Da(S_f(edge_f(1,1)))+Di(edge_f(1,2),1)+2)/(twoL+2))^2;
        %                 Q2 = max([Q2_ijmerge Q2_imergej Q2_jmergei]);
        Q2 = sum(Qa_2([1:type_min-1 type_min+1:type_max-1 type_max+1:end],1)) + Qab_a(S_f(edge_f(1,1)),S_f(edge_f(1,2)));
    end
    dQ_f = (Q2 - Q1)*1.0 + rand(1)/(1000*N);

    if dQ_t > dQ_f
        TP_aux = TP_aux + 1;
        results = [results;edge_t edge_f dQ_t dQ_f 1];
    elseif dQ_t == dQ_f
        if randi(2)==1
            TP_aux = TP_aux + 1;
            results = [results;edge_t edge_f dQ_t dQ_f 1];
        else
            results = [results;edge_t edge_f dQ_t dQ_f 0];
        end
    else
        results = [results;edge_t edge_f dQ_t dQ_f 0];
    end
end
AUC = TP_aux/Nsamples;
if net == 0
    results_foldername = sprintf('./LP_Mod_ResultsSTST/');
    if exist(results_foldername,'dir') ~= 7
        mkdir(results_foldername)
    end
    fid = fopen(sprintf('%sNM_LP_%s_%s.txt',results_foldername, name, samp),'w');
    fprintf(fid,'%f\n',AUC);
    fclose(fid);
else
    results_foldername = sprintf('./LP_Mod_ResultsSTST/');
    fid = fopen(sprintf('%sNM_LP_%s_%s.txt',results_foldername, name, samp),'a');
    fprintf(fid,'%f\n',AUC);
    fclose(fid);
end
dlmwrite(sprintf("%sresultsQ_%s_%s.txt",results_foldername,name, samp),results);

auc_score = AUC;

end

