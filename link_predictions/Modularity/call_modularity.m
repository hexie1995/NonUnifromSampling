addpath(genpath('/home2/xhe/modularity/'));
%path = 'C:\Users\hexie\OneDrive\Desktop\sampling\data\';
%savepath = 'C:\Users\hexie\OneDrive\Desktop\sampling\AUC_results\';

path = '/home/xhe/updated_edges_revised/';
savepath = '/home/xhe/MODULARITY_5_RUN/modularity_1/';

sampling_methods = {"RandomNodeSampler"; "DegreeBasedSampler"; "PageRankBasedSampler" ;"RandomEdgeSampler";
    "RandomNodeEdgeSampler" ;"HybridNodeEdgeSampler";"RandomEdgeSamplerWithInduction" ;"DiffusionSampler";"ForestFireSampler";
    "NonBackTrackingRandomWalkSampler"; "LoopErasedRandomWalkSampler";
    "RandomWalkSampler"; "RandomWalkWithRestartSampler"; "MetropolisHastingsRandomWalkSampler";
"CirculatedNeighborsRandomWalkSampler"; "BreadthFirstSearchSampler";"DepthFirstSearchSampler"; 
    "RandomWalkWithJumpSampler"; "RandomNodeNeighborSampler"; "ShortestPathSampler"};

for net  = 1:572
    netme = net -1;
    modularity_auc = []
    for k = 1:length(sampling_methods)
       samp = sampling_methods{k}
       try
           auc = modularity_for_sampling(path, netme, samp);
       catch
           auc = 0;
       end
       modularity_auc(end+1) = auc;

    end
    savename = sprintf('net%d__modularity_auc.txt',netme);
    savemything = sprintf('%s%s',savepath, savename);
    writematrix(modularity_auc, savemything);
end
