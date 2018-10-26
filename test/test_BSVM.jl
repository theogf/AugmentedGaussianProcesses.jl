using Distributions
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
println("Testing the BSVM model")
### TESTING WITH TOY XOR DATASET
    N_data = 1000;N_test = 20
    N_indpoints = 20; N_dim = 2
    noise = 1.0
    minx=-5.0; maxx=5.0
    function latent(x)
        return x[:,1].*sin.(1.0*x[:,2])
    end
    X = rand(N_data,N_dim)*(maxx-minx).+minx
    x1_test = range(minx,stop=maxx,length=N_test)
    x2_test = range(minx,stop=maxx,length=N_test)
    X_test = hcat([j for i in x1_test, j in x2_test][:],[i for i in x1_test, j in x2_test][:])
    y = sign.(latent(X)+rand(Normal(0,noise),size(X,1)))
    y_test = sign.(latent(X_test)+rand(Normal(0,noise),size(X_test,1)))


### TESTING WITH BANANA DATASET ###
    # X=readdlm("data/banana_X_train")
    # y=readdlm("data/banana_Y_train")[:]
    # maxs = [3.65,3.4]
    # mins = [-3.25,-2.85]
    # N_test = 40
    # x1_test = range(mins[1],stop=maxs[1],length=N_test)
    # x2_test = range(mins[2],stop=maxs[2],length=N_test)
    # X_test = hcat([j for i in x1_test, j in x2_test][:],[i for i in x1_test, j in x2_test][:])
    # y_test = ones(size(X_test,1))


kernel = AugmentedGaussianProcesses.RBFKernel([1.0],dim=N_dim)
autotuning=true
fullm = true
sparsem = true
stochm = true

ps = []; t_full = 0; t_sparse = 0; t_stoch =0;
#### FULL MODEL EVALUATION ####
if fullm
    println("Testing the full batch model")
    t_full = @elapsed fullmodel = AugmentedGaussianProcesses.BatchBSVM(X,y,Autotuning=autotuning,noise=noise,kernel=kernel,verbose=verbose)
    t_full += @elapsed fullmodel.train(iterations=20)
    y_full = fullmodel.predictproba(X_test); acc_full = 1-sum(abs.(sign.(y_full.-0.5)-y_test))/(2*length(y_test))
    if doPlots
        p1=plot(x1_test,x2_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=false,clims=(0,1),lab="",title="BSVM")
        push!(ps,p1)
    end
end

# #### SPARSE MODEL EVALUATION ####
if sparsem
    println("Testing the sparse model")
    t_sparse = @elapsed sparsemodel = AugmentedGaussianProcesses.SparseBSVM(X,y,Stochastic=false,Autotuning=autotuning,verbose=verbose,m=40,noise=noise,kernel=kernel,OptimizeIndPoints=true)
    t_sparse += @elapsed sparsemodel.train(iterations=1000)
    y_sparse = sparsemodel.predictproba(X_test); acc_sparse = 1-sum(abs.(sign.(y_sparse.-0.5)-y_test))/(2*length(y_test))
    if doPlots
        p2=plot(x1_test,x2_test,reshape(y_sparse,N_test,N_test),t=:contour,fill=true,cbar=false,clims=(0,1),lab="",title="Sparse BSVM")
        plot!(sparsemodel.inducingPoints[:,1],sparsemodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
        push!(ps,p2)
    end
end
# #### STOCH. MODEL EVALUATION ####
if stochm
    println("Testing the sparse stochastic model")
    t_stoch = @elapsed stochmodel = AugmentedGaussianProcesses.SparseBSVM(X,y,Stochastic=true,batchsize=10,Autotuning=autotuning,verbose=verbose,m=20,noise=noise,kernel=kernel,OptimizeIndPoints=true)
    t_stoch += @elapsed stochmodel.train(iterations=1000)
    y_stoch = stochmodel.predictproba(X_test); acc_stoch = 1-sum(abs.(sign.(y_stoch.-0.5)-y_test))/(2*length(y_test))
    if doPlots
        p3=plot(x1_test,x2_test,reshape(y_stoch,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(0,1),lab="",title="Stoch. Sparse BSVM")
        plot!(stochmodel.inducingPoints[:,1],stochmodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
        push!(ps,p3)
    end
end
t_full != 0 ? println("Full model : Acc=$(acc_full), time=$t_full") : nothing
t_sparse != 0 ? println("Sparse model : Acc=$(acc_sparse), time=$t_sparse") : nothing
t_stoch != 0 ? println("Stoch. Sparse model : Acc=$(acc_stoch), time=$t_stoch") : nothing

### Plot the computed predictions ###
if doPlots
    ptrue = plot(x1_test,x2_test,reshape(y_test,N_test,N_test),t=:contour,cbar=false,fill=:true)
    plot!(X[y.==1,1],X[y.==1,2],color=:red,t=:scatter,lab="y=1",title="Truth",xlims=(-5,5),ylims=(-5,5))
    plot!(X[y.==-1,1],X[y.==-1,2],color=:blue,t=:scatter,lab="y=-1")
    display(plot(ptrue,ps...));
end
return true
