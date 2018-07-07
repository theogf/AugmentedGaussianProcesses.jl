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

function RMSE(y_pred,y_test)
    return norm(y_pred-y_test)/sqrt(length(y_test))
end
dim = 1
n = 10000
N_test= 1000
X,f = generate_random_walk_data(n,dim,0.1)
X = (X-mean(X))/sqrt(var(X))
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
kernel = OMGP.RBFKernel(0.1)
##Basic Offline KMeans
t_off = @elapsed offgp = OMGP.SparseGPRegression(X,f,m=k,Stochastic=true,BatchSize=b,VerboseLevel=0,kernel=kernel)
t_off += @elapsed offgp.train(iterations=500)
y_off = offgp.predict(X_test)
y_indoff = offgp.predict(offgp.inducingPoints)
p1 = plot(X[:,1],f,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p1 = plot!(offgp.inducingPoints[:,1],y_indoff,t=:scatter,lab="",color=:red)
p1 = plot!(X_test,y_off,lab="",title="Offline KMeans")
println("Offline KMeans, RMSE : $(RMSE(offgp.predict(X),f)) in $t_off s")
###Online KMeans with Webscale
t_web = @elapsed onwebgp = OMGP.OnlineGPRegression(X,f,kmeansalg=OMGP.Webscale(),m=k,BatchSize=b,VerboseLevel=0,kernel=kernel)
t_web = @elapsed onwebgp.train(iterations=500)
y_web = onwebgp.predict(X_test)
y_indweb = onwebgp.predict(onwebgp.kmeansalg.centers)
p2 = plot(X[:,1],f,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p2 = plot!(onwebgp.kmeansalg.centers[:,1],y_indweb,t=:scatter,lab="",color=:red)
p2 = plot!(X_test,y_web,lab="",title="Webscale KMeans")
println("Webscale KMeans, RMSE : $(RMSE(onwebgp.predict(X),f)) in $t_web s")


###Online KMeans with Streaming
t_str = @elapsed onstrgp = OMGP.OnlineGPRegression(X,f,kmeansalg=OMGP.StreamOnline(),m=k,BatchSize=b,VerboseLevel=0,kernel=kernel)
t_str = @elapsed onstrgp.train(iterations=500)
y_str = onstrgp.predict(X_test)
y_indstr = onstrgp.predict(onstrgp.kmeansalg.centers)
p3 = plot(X[:,1],f,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p3 = plot!(onstrgp.kmeansalg.centers[:,1],y_indstr,t=:scatter,lab="",color=:red)
p3 = plot!(X_test,y_str,lab="",title="Streaming KMeans")
println("Streaming KMeans, RMSE : $(RMSE(onstrgp.predict(X),f)) in $t_str s")

display(plot(p1,p2,p3))
