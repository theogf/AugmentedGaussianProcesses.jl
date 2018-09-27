using Distributions
using LinearAlgebra
using Profile, ProfileView
# using Plots
# pyplot()
using OMGP
doPlots = false

println("Testing the XGPC")
### TESTING WITH TOY XOR DATASET
    N_data = 1000
    N_test = 40
    N_indpoints = 20
    N_dim = 2
    noise = 0.2
    minx=-5.0
    maxx=5.0
    function latent(x)
        return x[:,1].*sin.(0.5*x[:,2])
    end

    X = rand(N_data,N_dim)*(maxx-minx).+minx
    x_test = range(minx,stop=maxx,length=N_test)
    X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
    y = sign.(latent(X)+rand(Normal(0,noise),size(X,1)))
    y_test = sign.(latent(X_test)+rand(Normal(0,noise),size(X_test,1)))

### TESTING WITH BANANA DATASET ###
    # X=readdlm("data/banana_X_train")
    # y=readdlm("data/banana_Y_train")[:]
    # maxs = [3.65,3.4]
    # mins = [-3.25,-2.85]
    # x1_test = range(mins[1],maxs[1],N_test)
    # x2_test = range(mins[2],maxs[2],N_test)
    # X_test = hcat([j for i in x1_test, j in x2_test][:],[i for i in x1_test, j in x2_test][:])
    # y_test = ones(size(X_test,1))

(nSamples,nFeatures) = (N_data,1)
kernel = OMGP.ARDKernel([3.0],dim=N_dim)
stochmodel = OMGP.SparseXGPC(X,y,Stochastic=true,batchsize=40,Autotuning=true,AutotuningFrequency=2,VerboseLevel=2,m=N_indpoints,noise=noise,kernel=kernel,OptimizeIndPoints=false)
stochmodel.train(iterations=7)#,callback=savelog)
Profile.clear()
Profile.init()
@profile stochmodel.train(iterations=10)#,callback=savelog)
@time stochmodel.train(iterations=1000)
ProfileView.view()

println("blah")
# ps = []; t_full = 0; t_sparse = 0; t_stoch = 0;
# # #### FULL MODEL EVALUATION ####
# println("Testing the full model")
# t_full = @elapsed fullmodel = OMGP.BatchXGPC(X,y,noise=noise,kernel=kernel,VerboseLevel=3)
# t_full += @elapsed fullmodel.train(iterations=10)
# y_full = fullmodel.predictproba(X_test); acc_full = 1-sum(abs.(sign.(y_full.-0.5)-y_test))/(2*length(y_test))
# if doPlots
#     p1=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(0,1),lab="",title="XGPC")
#     push!(ps,p1)
# end
# # #### SPARSE MODEL EVALUATION ####
# println("Testing the sparse model")
# t_sparse = @elapsed sparsemodel = OMGP.SparseXGPC(X,y,Stochastic=false,Autotuning=true,ϵ=1e-6,VerboseLevel=3,m=N_indpoints,noise=noise,kernel=kernel,OptimizeIndPoints=false)
# metrics,savelog = OMGP.getLog(sparsemodel,X_test=X_test,y_test=y_test)
# t_sparse += @elapsed sparsemodel.train(iterations=100)#,callback=savelog)
# y_sparse = sparsemodel.predictproba(X_test); acc_sparse = 1-sum(abs.(sign.(y_sparse.-0.5)-y_test))/(2*length(y_test))
# if doPlots
#     p2=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,fill=true,cbar=false,clims=(0,1),lab="",title="Sparse XGPC")
#     plot!(sparsemodel.inducingPoints[:,1],sparsemodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
#     push!(ps,p2)
# end
# #### STOCH. SPARSE MODEL EVALUATION ###.
# println("Testing the sparse stochastic model")
# t_stoch = @elapsed stochmodel = OMGP.SparseXGPC(X,y,Stochastic=true,BatchSize=40,Autotuning=true,VerboseLevel=2,m=N_indpoints,noise=noise,kernel=kernel,OptimizeIndPoints=false)
# metrics,savelog = OMGP.getLog(stochmodel,X_test=X_test,y_test=y_test)
# t_stoch += @elapsed stochmodel.train(iterations=1000)#,callback=savelog)
# y_stoch = stochmodel.predictproba(X_test); acc_stoch = 1-sum(abs.(sign.(y_stoch.-0.5)-y_test))/(2*length(y_test))
# if doPlots
#     p3=plot(x_test,x_test,reshape(y_stoch,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(0,1),lab="",title="Stoch. Sparse XGPC")
#     plot!(stochmodel.inducingPoints[:,1],stochmodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
#     push!(ps,p3)
# end
#
# #### RESULTS OF THE ACCURACY ####
# t_full!=0 ? println("Full model : Acc=$(acc_full), time=$t_full s") : nothing
# t_sparse!=0 ? println("Sparse model : Acc=$(acc_sparse), time=$t_sparse s") : nothing
# t_stoch!=0 ? println("Stoch. Sparse model : Acc=$(acc_stoch), time=$t_stoch s") : nothing
#
# #### PRINTING RESULTS ####
# if doPlots
#     ptrue = plot(x_test,x_test,reshape(y_test,N_test,N_test),t=:contour,cbar=false,fill=:true)
#     plot!(X[y.==1,1],X[y.==1,2],color=:red,t=:scatter,lab="y=1",title="Truth",xlims=(-5,5),ylims=(-5,5))
#     plot!(X[y.==-1,1],X[y.==-1,2],color=:blue,t=:scatter,lab="y=-1")
#     display(plot(ptrue,ps...));
# end
# return true
