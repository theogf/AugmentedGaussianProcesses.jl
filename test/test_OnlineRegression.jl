using Distributions
using Plots
using Clustering
pyplot()
include("../src/OMGP.jl")
import OMGP

function generate_random_walk_data(N,dim,noise)
    X = zeros(N,dim)
    f = zeros(N)
    df = Normal(0,0.01)
    if dim == 1
        d = Normal(0,noise)
    else
        d = MvNormal(zeros(dim),noise)
    end
    for i in 2:N
        X[i,:] = X[i-1,:]+abs(rand(d))
        f[i] = f[i-1]+rand(df)
    end
    return X,f
end
dim = 1
n = 1000
N_test= 1000
X,f = generate_random_walk_data(n,dim,0.1)
X_test = linspace(minimum(X[:,1]),maximum(X[:,1]),N_test)

k = 100
b = 10
dorand = false
if dorand
    println("Randomizing the $n points first")
    randord = shuffle(1:n)
    X = X[randord,:]
    f = f[randord]
end
kernel = OMGP.RBFKernel(1.0)
###Basic Offline KMeans
offgp = OMGP.SparseGPRegression(X,f,m=k,Stochastic=true,BatchSize=b,VerboseLevel=2,kernel=kernel)
offgp.train(iterations=500)
y_off = offgp.predict(X_test)
y_indoff = offgp.predict(offgp.inducingPoints)
plot(X[:,1],f,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
plot!(offgp.inducingPoints[:,1],y_indoff,t=:scatter,lab="",color=:red)
p1 = plot!(X_test,y_off,lab="")

###Online KMeans with Webscale
onwebgp = OMGP.OnlineGPRegression(X,f,kmeansalg=OMGP.Webscale(),m=k,BatchSize=b,VerboseLevel=2,kernel=kernel)
onwebgp.train(iterations=500)
y_web = onwebgp.predict(X_test)
y_indweb = onwebgp.predict(onwebgp.kmeansalg.centers)
plot(X[:,1],f,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
plot!(onwebgp.kmeansalg.centers[:,1],y_indweb,t=:scatter,lab="",color=:red)
p2 = plot!(X_test,y_web,lab="")
display(plot(p2))


display(plot(p1,p2))
