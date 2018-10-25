using Distributions
using OMGP
using LinearAlgebra
using Random: seed!
seed!(1234)
if !@isdefined doPlots
    doPlots = true
end
if !@isdefined verbose
    verbose = 3
end
if doPlots
    using Plots
    pyplot()
end
N_data = 200
N_test = 30
N_dim = 2
noise = 0.2
minx=-5.0
maxx=5.0
function latent(x)
    return x[:,1].*sin.(x[:,2])
end
ν=20.0
kernel = RBFKernel(2.0); m = 40
autotuning = false

X = rand(N_data,N_dim)*(maxx-minx).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = latent(X)+rand(TDist(ν),size(X,1))
y_test = latent(X_test)
(nSamples,nFeatures) = (N_data,1)
ps = []; t_full = 0; t_sparse = 0; t_stoch = 0;

fullm = true
sparsem =true
stochm = true

if fullm
    println("Testing the full model")
    t_full = @elapsed global fullmodel = OMGP.BatchStudentT(X,y,noise=noise,kernel=kernel,verbose=verbose,Autotuning=autotuning,ν=ν)
    t_full += @elapsed fullmodel.train(iterations=100)
    y_full = fullmodel.predict(X_test); rmse_full = norm(y_full-y_test,2)/sqrt(length(y_test))
    if doPlots
        p1=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="StudentT")
        push!(ps,p1)
    end
end

if sparsem
    println("Testing the sparse model")
    t_sparse = @elapsed global sparsemodel = OMGP.SparseStudentT(X,y,Stochastic=false,Autotuning=autotuning,verbose=verbose,m=m,noise=noise,kernel=kernel,ν=ν)
    t_sparse += @elapsed sparsemodel.train(iterations=1000)
    y_sparse = sparsemodel.predict(X_test); rmse_sparse = norm(y_sparse-y_test,2)/sqrt(length(y_test))
    if doPlots
        p2=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="Sparse StudentT")
        plot!(sparsemodel.inducingPoints[:,1],sparsemodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
        push!(ps,p2)
    end
end

if stochm
    println("Testing the sparse stochastic model")
    t_stoch = @elapsed stochmodel = OMGP.SparseStudentT(X,y,Stochastic=true,batchsize=20,Autotuning=autotuning,verbose=verbose,m=m,noise=noise,kernel=kernel,ν=ν)
    t_stoch += @elapsed stochmodel.train(iterations=1000)
    y_stoch = stochmodel.predict(X_test); rmse_stoch = norm(y_stoch-y_test,2)/sqrt(length(y_test))
    if doPlots
        p3=plot(x_test,x_test,reshape(y_stoch,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(minx*1.1,maxx*1.1),lab="",title="Stoch. Sparse StudentT")
        plot!(stochmodel.inducingPoints[:,1],stochmodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
        push!(ps,p3)
    end
end
t_full != 0 ? println("Full model : RMSE=$(rmse_full), time=$t_full") : nothing
t_sparse != 0 ? println("Sparse model : RMSE=$(rmse_sparse), time=$t_sparse") : nothing
t_stoch != 0 ? println("Stoch. Sparse model : RMSE=$(rmse_stoch), time=$t_stoch") : nothing

if doPlots
    ptrue=plot(x_test,x_test,reshape(latent(X_test),N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="")
    plot!(ptrue,X[:,1],X[:,2],t=:scatter,lab="training points",title="Truth")
    display(plot(ptrue,ps...))
end

return true
