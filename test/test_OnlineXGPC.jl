using Distributions
using Plots
using Clustering
pyplot()
include("../src/OMGP.jl")
import OMGP



function generate_random_walk_data(N,dim,noise,monotone=true)
    X = zeros(N,dim)
    f = zeros(N)
    df = Normal(0,0.01)
    if dim == 1
        d = Normal(0,noise)
    else
        d = MvNormal(zeros(dim),noise)
    end
    for i in 2:N
        if monotone
            X[i,:] = X[i-1,:]+abs.(rand(d))
        else
            X[i,:] = X[i-1,:]+rand(d)
        end
        f[i] = f[i-1]+2*rand(df)
    end
    return X,f
end

function generate_uniform_data(N,dim,box_width)
    X = (rand(N,dim)-0.5)*box_width
end
function generate_gaussian_data(N,dim,variance=1.0)
    if dim == 1
        d = Normal(0,1)
    else
        d = MvNormal(zeros(dim),variance*Diagonal{Float64}(I,dim))
    end
    X = rand(d,N)
end

function plotting1D(X,f,ind_points,pred_ind,X_test,pred,title)
    p = plot(X[:,1],f,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
    p = plot!(ind_points[:,1],pred_ind,t=:scatter,lab="",color=:red)
    p =  plot!(X_test,pred,lab="",title=title)
    return p
end

function plotting2D(X,f,ind_points,pred_ind,x1_test,x2_test,pred,minf,maxf,title;full=false)
    N_test = size(x1_test,1)
    p = plot(x1_test,x2_test,reshape(pred,N_test,N_test),t=:contour,fill=true,lab="",title=title)
    p = plot!(X[:,1],X[:,2],zcolor=f,t=:scatter,lab="",alpha=0.8,markerstrokewidth=0)
    if !full
        p = plot!(ind_points[:,1],ind_points[:,2],zcolor=pred_ind,t=:scatter,lab="",color=:red)
    end
    return p
end

function plottingtruth(X,f,X_test,x1_test,x2_test)
    N_test = size(x1_test,1)
    true_f = randomf(X_test)
    p = plot(x1_test,x2_test,reshape(true_f,N_test,N_test),t=:contour,fill=true,lab="",title="Truth")
    p = plot!(X[:,1],X[:,2],zcolor=f,t=:scatter,lab="",alpha=0.8,markerstrokewidth=0)
end
function loss(y_pred,y_test)
    return 0.5*sum(1-y_pred.*sign.(y_test-0.5))/length(y_test)
end

function randomf(X,option=1)
    # return X[:,1].^2+cos.(X[:,1]).*sin.(X[:,2])-sin.(X[:,1]).*tanh.(X[:,2])-X[:,1]
    return X[:,1].*sin.(X[:,2])
end
dim = 2
monotone = false
n = 1000
N_test= 10
noise=0.01
# X,f = generate_random_walk_data(n,dim,0.1,monotone)
X = generate_uniform_data(n,dim,1)
# X = generate_gaussian_data(n,dim)'
if dim == 2
    f = randomf(X)+rand(Normal(0,noise),size(X,1))
end
X = (X-mean(X))/sqrt(var(X))
if dim == 1
    X_test = range(minimum(X[:,1]),maximum(X[:,1]),N_test)
elseif dim == 2
    x1_test = range(minimum(X[:,1]),maximum(X[:,1]),N_test)
    x2_test = range(minimum(X[:,2]),maximum(X[:,2]),N_test)
    X_test = hcat([j for i in x1_test, j in x2_test][:],[i for i in x1_test, j in x2_test][:])
end
k = 100
b = 10
dorand = true
if dorand
    println("Randomizing the $n points first")
    randord = shuffle(1:n)
    X = X[randord,:]
    f = f[randord]
end

y=sign.(f)
kernel = OMGP.RBFKernel(0.5)

##Basic Offline KMeans
t_off = @elapsed offgp = OMGP.SparseXGPC(X,f,m=k,Stochastic=true,Autotuning=false,BatchSize=b,VerboseLevel=0,kernel=kernel)
t_off += @elapsed offgp.train(iterations=500)
y_off = offgp.predict(X_test)
y_indoff = offgp.predict(offgp.inducingPoints)
if dim == 1
    p1 = plotting1D(X,y,offgp.inducingPoints,y_indoff,X_test,y_off,"OfflineKMeans")
elseif dim == 2
    p1 = plotting2D(X,y,offgp.inducingPoints,y_indoff,x1_test,x2_test,y_off,minf,maxf,"OfflineKMeans")
end
println("Offline KMeans ($t_off s)\n\tRMSE (train) : $(loss(offgp.predict(X),f))\n\tRMSE (test) : $(loss(y_off,randomf(X_test)))")

###Online KMeans with Webscale
t_web = @elapsed onwebgp = OMGP.OnlineXGPC(X,y,kmeansalg=OMGP.Webscale(),m=k,BatchSize=b,VerboseLevel=0,kernel=kernel)
t_web = @elapsed onwebgp.train(iterations=500)
y_web = onwebgp.predict(X_test)
y_indweb = onwebgp.predict(onwebgp.kmeansalg.centers)
if dim == 1
    p2 = plotting1D(X,f,onwebgp.kmeansalg.centers,y_indweb,X_test,y_web,"Webscale KMeans (k=$(onwebgp.m))")
elseif dim == 2
    p2 = plotting2D(X,f,onwebgp.kmeansalg.centers,y_indweb,x1_test,x2_test,y_web,minf,maxf,"Webscale KMeans (k=$(onwebgp.m))")
end
println("Webscale KMeans ($t_web s)\n\tRMSE (train) : $(loss(onwebgp.predict(X),f))\n\tRMSE (test) : $(loss(y_web,randomf(X_test)))")


###Online KMeans with Streaming
t_str = @elapsed onstrgp = OMGP.OnlineXGPC(X,f,kmeansalg=OMGP.StreamOnline(),m=k,BatchSize=b,VerboseLevel=0,kernel=kernel)
t_str = @elapsed onstrgp.train(iterations=500)
y_str = onstrgp.predict(X_test)
y_indstr = onstrgp.predict(onstrgp.kmeansalg.centers)



if dim == 1
    p3 = plotting1D(X,f,onstrgp.kmeansalg.centers,y_indstr,X_test,y_str,"Streaming KMeans (m=$(onstrgp.m))")
elseif dim == 2
    p3 = plotting2D(X,f,onstrgp.kmeansalg.centers,y_indstr,x1_test,x2_test,y_str,minf,maxf,"Streaming KMeans (m=$(onstrgp.m))")
end
println("Streaming KMeans ($t_str s)\n\tRMSE (train) : $(loss(onstrgp.predict(X),f))\n\tRMSE (test) : $(loss(y_str,randomf(X_test)))")



### Non sparse GP :
t_full = @elapsed fullgp = OMGP.BatchXGPC(X,y,kernel=kernel)
t_full = @elapsed fullgp.train()
y_full = fullgp.predict(X_test)
if dim == 1
    p4 = plotting1D(X,f,[0],[0],y_full,"Full batch GP",full=true)
elseif dim == 2
    p4 = plotting2D(X,f,[0 0],0,x1_test,x2_test,y_full,minf,maxf,"Full batch GP",full=true)
end
println("Full GP ($t_web s)\n\tRMSE (train) : $(loss(fullgp.predict(X),f))\n\tRMSE (test) : $(loss(y_full,randomf(X_test)))")

if dim == 2
    p = plottingtruth(X,f,X_test,x1_test,x2_test)
    display(plot(p,p4,p1,p2,p3)); gui()
else
    display(plot(p1,p2,p3)); gui()
end

# model = OMGP.GPRegression(X,f,kernel=kernel)
#
#
# display(plotting2D(X,f,onstrgp.kmeansalg.centers,y_indstr,x1_test,x2_test,y_pred,"Streaming KMeans"))
# display(plotting1D(X,f,onstrgp.kmeansalg.centers,y_indstr,X_test,y_pred,"Streaming KMeans"))
