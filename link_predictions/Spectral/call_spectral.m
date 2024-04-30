addpath(genpath('/home/xhe/after_samp_process/link_predictions/Spectral/m1/'));
%path = 'C:\Users\hexie\OneDrive\Desktop\sampling\data\';
%savepath = 'C:\Users\hexie\OneDrive\Desktop\sampling\AUC_results\';

path1 = '/home/xhe/updated_edges_revised/';
path2 = '/home/xhe/sampled_edges_2024_old/';

savepath = '/home/xhe/after_samp_process/link_predictions/Spectral/m1/';



sampling_methods = {"RandomNodeSampler"; "DegreeBasedSampler"; "PageRankBasedSampler" ;"RandomEdgeSampler";
    "RandomNodeEdgeSampler" ;"HybridNodeEdgeSampler";"RandomEdgeSamplerWithInduction" ;"DiffusionSampler";"ForestFireSampler";
    "NonBackTrackingRandomWalkSampler"; "LoopErasedRandomWalkSampler";
    "RandomWalkSampler"; "RandomWalkWithRestartSampler"; "MetropolisHastingsRandomWalkSampler";
"CirculatedNeighborsRandomWalkSampler"; "BreadthFirstSearchSampler";"DepthFirstSearchSampler"; 
    "RandomWalkWithJumpSampler"; "RandomNodeNeighborSampler"; "ShortestPathSampler"};


for net  = 1:572
    netme = net -1;
    spectral_auc = []
    for k = 1:length(sampling_methods)
       samp = sampling_methods{k}
       try
           auc = spectral_for_sampling(path1, path2, netme, samp);
       catch
           auc = 0;
       end
       spectral_auc(end+1) = auc;
    end
    savename = sprintf('net%d__spectral_auc.txt',netme);
    savemything = sprintf('%s%s',savepath, savename);
    writematrix(spectral_auc, savemything);
end