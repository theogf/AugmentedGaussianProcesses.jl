# XGPC
Code Repository for paper : "Extrememely Fast Gaussian Process Classification via Polya-Gamma Data Augmentation"

You will find here the code for the Polya-Gamma Augmented Gaussian Process Classification in Julia as well as some light datasets.
This folder contains the algorithm, the code that produced the experiments showed in the submitted paper to AIStats 2018 and some datasets light enough for GitHub.
It is planned to be transformed as a Julia package in the near future for public use.
The framework may look weird at first but it is because it has been designed to handle other data augmented models.

# Tips for code exploration :

The '''Train''' function is implemented in **DataAugmentedModelFunctions.jl**

The Updates function of the XGPC are found in **XGPC_Functions.jl**

The methods for matrix computations, stochastic methods and prediction methods are in **ModelSpecificFunctions.jl**

It is relatively complex to link the GPFlow library to Julia, I would suggest to run the experiments with only the XGPC with the flags at the beginning of **paper_experiments.jl**

# Run the package

Get Julia 0.6.0, install packages (`Pkg.add('X')`) : Distributions, StatsBase, PyPlot, QuadGK.
Open **paper_experiments.jl** and set the flags for the methods you want to use :

    XGPC, SVGPC or Logistic Regression

SVGPC requires to install Tensorflow and  GPFlow on Python and PyCall on Julia.
Logistic Regression requires to install ScikitLearn on Julia

If you want to simply use XGPC you can also do:
```
include(DataAugmentedModels.jl)
model = SparseXGPC(X_train,Y_train;Stochastic=?,BatchSize=?,m=?,Kernels=?) #Parameters after ; are optional
model.train!(iterations=100)
Y_predic = model.predict(X_test)
```
