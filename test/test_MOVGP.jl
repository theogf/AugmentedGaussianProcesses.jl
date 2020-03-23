using AugmentedGaussianProcesses



X = [rand(10,3) for _ in 1:3]
y = [rand(10) for _ in 1:3]
m = AGP.MOVGP(X,y,SqExponentialKernel(),GaussianLikelihood(),AnalyticVI(),3)
m2 = VGP(X[1],y[1], SqExponentialKernel(),StudentTLikelihood(2.0),AnalyticVI())
train!(m, 10)
train!(m2, 10)
