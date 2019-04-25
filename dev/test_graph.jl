using LightGraphs, GraphPlot
using AugmentedGaussianProcesses
using DeterminantalPointProcesses
using LinearAlgebra
using BenchmarkTools
using Plots
N = 5000
X = rand(N,2)*5;
K = kernelmatrix(X,RBFKernel());
dpp = DeterminantalPointProcess(Symmetric(K+1e-7*I))
K[K.<0.5] .= 0;
g = LightGraphs.SimpleGraph(K)
graphset = LightGraphs.dominating_set(g,MinimalDominatingSet())
isindpoint = ones(Int64,N)
isindpoint[graphset] .= 2
nodecolor = [colorant"lightseagreen",colorant"orange"];
display(gplot(g,X[:,1],X[:,2],nodefillc=nodecolor[isindpoint]))
pgraph = scatter(eachcol(X)...,lab="");
scatter!(eachcol(X[graphset,:])...,lab="",title="k=$(length(graphset))")
dppset = rand(dpp,1)[1]
pdpp = scatter(eachcol(X)...,lab="");
scatter!(eachcol(X[dppset,:])...,lab="",title="k=$(length(dppset))")
display(plot(pgraph,pdpp))
logpmf(dpp,graphset)
logpmf(dpp,dppset)
K = kernelmatrix(X,RBFKernel());
@btime LightGraphs.dominating_set($g,MinimalDominatingSet());
@btime dppset = rand($dpp,1)[1]
