using Test
using AugmentedGaussianProcesses
using Statistics
using LinearAlgebra
const AGP = AugmentedGaussianProcesses
using HDF5
nData = 100; nDim = 2
m = 50; b= 10
k = AGP.RBFKernel()

X = rand(nData,nDim)
y =Dict("Regression"=>norm.(eachrow(X)),"Classification"=>Int64.(sign.(norm.(eachrow(X)).-0.5)),"MultiClass"=>floor.(Int64,norm.(eachrow(X.*2))))
using Plots

scatter(X[:,1],X[:,2],zcolor=y["Regression"])
scatter(X[:,1],X[:,2],zcolor=y["Classification"])
scatter(X[:,1],X[:,2],zcolor=y["MultiClass"])
