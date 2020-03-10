using AugmentedGaussianProcesses
using Distances
using Random
ν=3.0

testfunction(x) = (1+x/ν)^(-0.5*(ν+1))
testfunction(x) = sech(sqrt(x/2))
testfunction(x) = (1+sqrt(3*x))*exp(-sqrt(3*x))
testfunction(x) = (1+sqrt(5*x)+5/3*x)*exp(-sqrt(5*x))
testdist = AGP.LaplaceTransformDistribution(testfunction)

rand(testdist)
laptrans(testdist,n=1)[1]
p = histogram(rand(testdist,10000),bins=range(0,3,length=100),normalize=true)|>display
using Distributions
histogram!(rand(Gamma((ν+1)/2,1/ν),10000),alpha=0.5,bins=range(0,10,length=100),normalize=true)
histogram!([AugmentedGaussianProcesses.draw(AugmentedGaussianProcesses.PolyaGammaDist(),1,0) for i in 1:10000],alpha=0.5,bins=range(0,10,length=100),normalize=true)
