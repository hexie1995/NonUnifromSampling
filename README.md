# NonUnifromSampling
Non-uniform sampling for Link Preidction

</div>

<h2 align="center">System Requirements </h2>

To reproduce all results from our experiments, you will first need to install [LittleBallOfFur](https://github.com/benedekrozemberczki/littleballoffur) from their Github. 

Then you will also need to install the codes available by Lucashu in the [link-prediction](https://github.com/lucashu1/link-prediction) Github. 

To run the Stacking link prediction, please check the code available by [Amir Ghasemian et al.](https://github.com/Aghasemian/OptimalLinkPrediction) from their Github. 

The dataset we used in this paper could also be found in the Stacking link prediction Github repo. 

Once you install the python packages in the above Github in the same envrionment (or different if needed). All the codes above will run smoothly.

Also, to run the DCSBM or Spectral code, you need to install MATLAB. 



You can check your python version with

```bash
$ python --version
```

Note very importantly, the codes provided here are all for paralleization on a cluster, and will cause huge lag if run on a personal laptop. It is thus recommended that you have it run in a linux envirnment. 
