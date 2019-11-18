using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using KernelFunctions
using MLDataUtils
using Distributions, LinearAlgebra
using Random: seed!
using Plots; gr()


N = 1000; M = 20
x = collect(range(-5,5,length=N))
X = reshape(x,:,1)
k = SqExponentialKernel(1.0)
K = 5
ys = zeros(Int64,N)
fs = zeros(N,K)
while length(unique(ys)) != K
    fs .= rand(MvNormal(kernelpdmat(k,X,obsdim=1)),K)
    ys .= getindex.(findmax.(eachrow(fs)),2)
end
plot(X,fs)
scatter!(X,1.1*maximum(fs)*ones(N),color=ys,msw=0.0,levels=5) |> display
var_k= 50
smodel = SVGP(X,ys,k,LogisticSoftMaxLikelihood(),AnalyticSVI(50),M,optimizer=false,variance=var_k)
train!(smodel,100)

ysp_prob = proba_y(smodel,X)
ysp = predict_y(smodel,X)
nQ = 6
allymos = Vector(undef,nQ)
for Q in 2:(nQ+1)
    @info "Running Model with $(Q+1) latent GP"
    mosmodel = MOSVGP(X,[ys],k,LogisticSoftMaxLikelihood(),AnalyticSVI(50),Q,M,optimizer=false,variance=var_k)
    train!(mosmodel,200)
    # ymos1_prob = proba_y(mosmodel,X)[1]
    ymos = predict_y(mosmodel,X)[1]
    allymos[Q-1] = ymos
end
p = scatter(x,zero(x),color=ys,msw=0.0,lab="")
scatter!(x,ones(length(x)),color=ysp,msw=0.0,lab="")
for i in 1:nQ
    scatter!(x,(i+1)*ones(length(x)),color=allymos[i],msw=0.0,lab="")
end
p
