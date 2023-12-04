import numpy as np
import os
import os.path
#net462__3PageRankBasedSampler_t_train

path = r"/home2/xhe/updated_edges_revised//"


# +
sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'DiffusionTreeSampler', 'ForestFireSampler',
 'CommonNeighborAwareRandomWalkSampler','NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 
 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler','CommunityStructureExpansionSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

sampling_methods = ['RandomNodeSampler', 'DegreeBasedSampler', 'PageRankBasedSampler', 'RandomEdgeSampler',
 'RandomNodeEdgeSampler', 'HybridNodeEdgeSampler','RandomEdgeSamplerWithPartialInduction',
 'RandomEdgeSamplerWithInduction', 'DiffusionSampler',
 'ForestFireSampler',
 'NonBackTrackingRandomWalkSampler', 'LoopErasedRandomWalkSampler',
 'RandomWalkSampler', 'RandomWalkWithRestartSampler','MetropolisHastingsRandomWalkSampler', 'CirculatedNeighborsRandomWalkSampler', 'BreadthFirstSearchSampler',
 'DepthFirstSearchSampler', 'RandomWalkWithJumpSampler',
 'RandomNodeNeighborSampler', 'ShortestPathSampler']

do_not_touch = ['net10_', 'net128_','net12_', 'net13_','net147_', 'net18_','net22_','net282_','net298_','net307_','net308_',
                'net310_','net312_','net318_', 'net323_','net325_','net329_','net331_','net335_', 'net33_', 'net341_', 'net345_',
                'net347_','net348_','net349_', 'net34_','net357_','net361_', 'net362_','net364_','net36_','net37_','net38_','net397_',
                'net39_','net400_','net403_','net412_', 'net416_','net417_','net421_','net432_','net435_','net436_','net437_',
                'net440_','net443_','net453_','net45_','net513_','net515_','net568_','net570_','net571_','net79_','net8_', 'net9_']

# -

for count in range(1,6):
    unfinished_net = []
    unfinished_samp = []
    for i in range(572):
        for samp in sampling_methods:
            name = "net" + str(i) + "_"
            if name not in do_not_touch:
                fname = path + name + "_" + str(count) + samp + "_f_test_10000.npy"
                #print(fname)
                if not os.path.isfile(fname) :
                    unfinished_net.append(name)
                    unfinished_samp.append(samp)

    unfinished_net = np.array(unfinished_net)
    unfinished_samp = np.array(unfinished_samp)
    unfinished = np.vstack((unfinished_net,unfinished_samp))
    np.savetxt("unfinished" + str(count)+ ".txt", unfinished, fmt='%s')

