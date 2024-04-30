# Link Prediction Accuracy on Real-World Networks Under Non-Uniform Missing-Edge Patterns

This is the Github accompanying the paper "Link Prediction Accuracy on Real-World Networks Under
Non-Uniform Missing-Edge Patterns", submitted to PLOS One and available as Preprint on Arxiv at [preprint](https://arxiv.org/abs/2401.15140)

</div>

<h2 align="center">System Requirements </h2>

To reproduce all results from our experiments, you will first need to install the sampling codes from [LittleBallOfFur](https://github.com/benedekrozemberczki/littleballoffur). 

Then you will also need to install the codes available by Lucashu in their [link-prediction](https://github.com/lucashu1/link-prediction) Github. 
Note importantly, these codes, which include Node2Vec Embedding, Adamic-Adar, Preferential Attachment, Node2Vec Prod, and Jaccard Coefficient, can only be done in Python 2.7 or earlier.  

To run the Spectral Clustering, Modularity, MDL-DCSBM, and the Top-Stacking link prediction codes, please check the code available by [Amir Ghasemian et al.](https://github.com/Aghasemian/OptimalLinkPrediction) from their Github. 
Some of these codes were in MATLAB, and others were in Python 3.7 or later. 
The dataset we used in this paper could also be found both here and in [Amir Ghasemian et al.](https://github.com/Aghasemian/OptimalLinkPrediction). 

Once you install the python packages in the above Github in the same envrionment (or different if needed). 
Then proceed to run the codes in the order of:
1. Preprocess(which include the non-unifrom sampling part)
2. Link Prediction

You can check your python version with
```bash
$ python --version
```
Note very importantly, the codes provided here are all for paralleization on a cluster, and will cause huge lag if run on a personal laptop. It is thus recommended that you have it run in a linux envirnment with a cluster more than 50 cores. 
