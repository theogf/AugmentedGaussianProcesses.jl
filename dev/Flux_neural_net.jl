using Flux
using Zygote
using Plots
using Statistics, StatsBase, Distributions, LinearAlgebra

nDim = 2
nData = 200
foo(x,y) = y.*sin.(x)+x.*cos.(y)
logit(x) = log.(x/(10.0-x))
x_grid = range(-5,5,length=100)
contour(x_grid,x_grid,foo,fill=true)
X = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])
y = foo(X[:,1],X[:,2]) + 0.5*randn(size(X,1))


NN_μ = Chain(Dense(nDim,32,relu),Dense(32,1),logit)
NN_w = Chain(Dense(nDim,32,relu),Dense(32,1))
NN_l= Chain(Dense(nDim^2,32,relu),Dense(32,1),log)
X = copy(transpose(X))

loss(x,y) = norm(NN_w(x).-y)

opt = ADAM()
evalcb = () -> @show loss(X,y)
@epochs 5 begin; Flux.train!(loss,Flux.params(NN_w),[(X,y)],opt,cb=evalcb);
contour(x_grid,x_grid,reshape(NN_w(X),100,100));

NN_w(X)
