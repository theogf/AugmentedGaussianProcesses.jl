using Distributions
using LinearAlgebra

using AugmentedGaussianProcesses
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
println("Testing the XGPC model")
### TESTING WITH TOY XOR DATASET
    N_data = 100;N_test = 20
    N_indpoints = 50; N_dim = 2
    noise = 2.0
    minx=-5.0; maxx=5.0
    function latent(x)
        return x[:,1].*sin.(1.0*x[:,2])
    end
    X = rand(N_data,N_dim)*(maxx-minx).+minx
    x_test = range(minx,stop=maxx,length=N_test)
    X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
    y = sign.(latent(X)+rand(Normal(0,noise),size(X,1)))
    y_test = sign.(latent(X_test))

kernel = RBFKernel([3.0],dim=N_dim)
autotuning=true
optindpoints=false
fullm = true
sparsem = true
ssparsem = true
ps = []; t_full = 0; t_sparse = 0; t_stoch = 0;
# # #### FULL MODEL EVALUATION ####
if fullm
    println("Testing the full model")
    t_full = @elapsed fullmodel = AugmentedGaussianProcesses.BatchXGPC(X,y,noise=noise,kernel=kernel,verbose=verbose,Autotuning=autotuning)
    t_full += @elapsed fullmodel.train(iterations=20)
    _ =  fullmodel.predict(X_test)
    y_full = fullmodel.predictproba(X_test); acc_full = 1-sum(abs.(sign.(y_full.-0.5)-y_test))/(2*length(y_test))
    if doPlots
        p1=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(0,1),lab="",title="XGPC")
        push!(ps,p1)
    end
end
# # #### SPARSE MODEL EVALUATION ####
if sparsem
    println("Testing the sparse model")
    t_sparse = @elapsed sparsemodel = AugmentedGaussianProcesses.SparseXGPC(X,y,Stochastic=false,Autotuning=autotuning,ϵ=1e-6,verbose=verbose,m=N_indpoints,noise=1e-3,kernel=kernel,OptimizeIndPoints=optindpoints)
    t_sparse += @elapsed sparsemodel.train(iterations=100)
    _ =  sparsemodel.predict(X_test)
    y_sparse = sparsemodel.predictproba(X_test); acc_sparse = 1-sum(abs.(sign.(y_sparse.-0.5)-y_test))/(2*length(y_test))
    if doPlots
        p2=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,fill=true,cbar=false,clims=(0,1),lab="",title="Sparse XGPC")
        plot!(sparsemodel.inducingPoints[:,1],sparsemodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
        push!(ps,p2)
    end
end

#### STOCH. SPARSE MODEL EVALUATION ###.
if ssparsem
    println("Testing the sparse stochastic model")
    t_stoch = @elapsed stochmodel = AugmentedGaussianProcesses.SparseXGPC(X,y,Stochastic=true,batchsize=40,Autotuning=autotuning,verbose=verbose,m=N_indpoints,noise=noise,kernel=kernel,OptimizeIndPoints=optindpoints)
    t_stoch += @elapsed stochmodel.train(iterations=500)
    _ =  stochmodel.predict(X_test)
    y_stoch = stochmodel.predictproba(X_test); acc_stoch = 1-sum(abs.(sign.(y_stoch.-0.5)-y_test))/(2*length(y_test))
    if doPlots
        p3=plot(x_test,x_test,reshape(y_stoch,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(0,1),lab="",title="Stoch. Sparse XGPC")
        plot!(stochmodel.inducingPoints[:,1],stochmodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
        push!(ps,p3)
    end
end
#
#### RESULTS OF THE ACCURACY ####
t_full!=0 ? println("Full model : Acc=$(acc_full), time=$t_full s") : nothing
t_sparse!=0 ? println("Sparse model : Acc=$(acc_sparse), time=$t_sparse s") : nothing
t_stoch!=0 ? println("Stoch. Sparse model : Acc=$(acc_stoch), time=$t_stoch s") : nothing
#
#### PRINTING RESULTS ####
if doPlots
    ptrue = plot(x_test,x_test,reshape(y_test,N_test,N_test),t=:contour,cbar=false,fill=:true)
    plot!(X[y.==1,1],X[y.==1,2],color=:red,t=:scatter,lab="y=1",title="Truth",xlims=(-5,5),ylims=(-5,5))
    plot!(X[y.==-1,1],X[y.==-1,2],color=:blue,t=:scatter,lab="y=-1")
    display(plot(ptrue,ps...));
end
return true
