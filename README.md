# XGPC
<!--Code Repository for paper : "Extrememely Fast Gaussian Process Classification via Polya-Gamma Data Augmentation"

You will find here the code for the Polya-Gamma Augmented Gaussian Process Classification in Julia as well as some light datasets.
This folder contains the algorithm, the code that produced the experiments showed in the submitted paper to AIStats 2018 and some datasets light enough for GitHub.
It is planned to be transformed as a Julia package in the near future for public use.
The framework may look weird at first but it is because it has been designed to handle other data augmented models.-->

# Tips for code exploration :

The '''Train''' function is implemented in **DataAugmentedModelFunctions.jl**

The Updates function of the XGPC are found in **XGPC_Functions.jl**

The methods for matrix computations, stochastic methods and prediction methods are in **ModelSpecificFunctions.jl**


# Run the package

Get Julia > 0.6.0, install packages (`Pkg.add('X')`) : Distributions, StatsBase, PyPlot, QuadGK, Clustering.

```
include("DataAugmentedModels.jl")
using DAM
model = SparseXGPC(X_train,Y_train;Stochastic=?,BatchSize=?,m=?,Kernels=?) #Parameters after ; are optional
model.train(iterations=100)
Y_predic = model.predict(X_test) #For getting the label directly
Y_predic_prob = model.predictproba(X_test) #For getting the likelihood of predicting class 1
```
