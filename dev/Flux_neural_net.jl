using Flux
using Zygote
using Plots
using Statistics, StatsBase, Distributions, LinearAlgebra
using Base.Iterators: partition
using MLDataUtils

nDim = 2
nData = 200
foo(x,y) = y.*sin.(x)+x.*cos.(y)
logit(x) = log.(x/(10.0-x))
x_grid = range(-5,5,length=100)
contour(x_grid,x_grid,foo,fill=true)
X = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])
Y = foo(X[:,1],X[:,2]) + 0.5*randn(size(X,1))


NN_μ = Chain(Dense(nDim,32,relu),Dense(32,1),logit)
NN_w = Chain(Dense(nDim,32,relu),Dense(32,1))
NN_l= Chain(Dense(nDim^2,32,relu),Dense(32,1),log)
X = copy(transpose(X))

data = shuffleobs((X,Y))

loss(x,y) = norm(NN_w(x).-y)^2
collect(eachbatch((X,Y),size=5))
opt = ADAM()
evalcb = () -> @show loss(X,Y)
Flux.train!(loss,Flux.params(NN_w),eachbatch(data,size=100),opt,cb=evalcb)
contour(x_grid,x_grid,reshape(NN_w(X).data,100,100),fill=true)


Flux.params(NN_w).params
