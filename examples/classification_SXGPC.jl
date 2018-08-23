using Plots
using GradDescent
pyplot()
#unicodeplots()
using Distributions
using ValueHistories
using OMGP
using ProfileView
doPlot = false
use_dataset = false
X_data = Array{Float64,2}(0,0)
if use_dataset
    (X_data,y_data,DatasetName) = get_Dataset("German")
    (nSamples,nFeatures) = size(X_data);

    nFold = 10; #Chose the number of folds
    fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
    #Global variables for debugging
    X = []; y = []; X_test = []; y_test = [];i=4
    X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
    y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
    X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
    y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]

else
    N_data = 100
    N_test = 100
    noise = 0.5
    function latent(x)
        return sin.(2*x)
    end
    X_data = reshape(sort(rand(N_data))*10.0,:,1)
    X_test = collect(range(0,stop=10.0,length=N_test))
    y_data = sign.(1.0./(1.0+exp.(-latent(X_data)+rand(Normal(0,noise),length(X_data))))-0.5)
    y_test = sign.(1.0./(1.0+exp.(-latent(X_test)+rand(Normal(0,noise),length(X_test))))-0.5)
    X=X_data; y=y_data
    (nSamples,nFeatures) = (N_data,1)
end

MaxIter = 500#Maximum number of iterations for every algorithm
M=100; θ=5.0; ϵ=1e-4; noise=1e-3
kerns = [Kernel("rbf",1.0;params=θ)]
# kerns = [Kernel("rbf",1.0;params=θ);Kernel("linear",1.0)]
 # kerns = [Kernel("linear",1.0)]
batchsize = 30
Ninducingpoints = 20

 # toc()
 tic()
# model = DAM.BatchXGPC(X,y;Kernels=kerns,Autotuning=true,optimizer=StandardGD(α=0.1),AutotuningFrequency=1,VerboseLevel=2,ϵ=1e-4,nEpochs=100)
Profile.clear()
@profile model = SparseXGPC(X,y;optimizer=Adam(α=0.5),OptimizeIndPoints=true,
Stochastic=false,ϵ=1e-4,nEpochs=MaxIter,SmoothingWindow=10,Kernels=kerns,Autotuning=false,AutotuningFrequency=2,
VerboseLevel=2,AdaptiveLearningRate=true,batchsize=batchsize,m=Ninducingpoints)
ProfileView.view()
initPoints = copy(model.inducingPoints)
# iter_points = vcat(collect(1:1:9),collect(10:10:99))
iter_points = collect(1:1:1000)
metrics = MVHistory()
Parameters = MVHistory()
function StoreIt(model::OMGP.GPModel,iter;hyper=false)#;iter_points=[],LogArrays=[],X_test=0,y_test=0)
    if in(iter,iter_points)
        if !hyper
            push!(metrics,:time_init,iter,time_ns()*1e-9)
            y_p = model.PredictProba(X_test)
            loglike = zeros(y_p)
            loglike[y_test.==1] = log.(y_p[y_test.==1])
            loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
            push!(metrics,:accuracy,iter,1-sum(1-y_test.*sign.(y_p-0.5))/(2*length(y_test)))
            push!(metrics,:meanloglikelihood,iter,mean(loglike))
            push!(metrics,:medianloglikelihood,iter,median(loglike))
            push!(metrics,:ELBO,iter,DAM.ELBO(model))
            push!(metrics,:end_time,iter,time_ns()*1e-9)
            println("Iteration $iter : Accuracy is $(1-sum(1-y_test.*sign.(y_p-0.5))/(2*length(y_test))), ELBO is $(DAM.ELBO(model)), θ is $(model.Kernels[1].param)")
            push!(Parameters,:μ,iter,model.μ)
            push!(Parameters,:diag_Σ,iter,diag(model.Σ))
            push!(Parameters,:kernel_params,iter,getfield.(model.Kernels,:param))
            push!(Parameters,:kernel_coeffs,iter,getfield.(model.Kernels,:coeff))
        else
            push!(metrics,:ELBO_posthyper,iter,DAM.ELBO(model))
        end
    end
end
# model = SparseBSVM(X,y;Stochastic=true,Kernels=kerns,Autotuning=true,SmoothingWindow=50,AutotuningFrequency=1,VerboseLevel=3,ρ_AT=0.2,AdaptiveLearningRate=true,BatchSize=50,m=50)
# model = BatchBSVM(X,y;Kernels=kerns,Autotuning=false,AutotuningFrequency=2,VerboseLevel=1)
 # model = LinearBSVM(X,y;Intercept=true,Stochastic=false,BatchSize=30,AdaptiveLearningRate=true,VerboseLevel=3,Autotuning=true,AutotuningFrequency=5)
Profile.clear()
@profile model.train()
ProfileView.view()
 # model.train(callback=StoreIt,convergence=function (model::AugmentedClassifier,iter)  return LogLikeConvergence(model,iter,X_test,y_test);end)
y_predic_log = model.predict(X_test)
println(1-sum(1-y_test.*y_predic_log)/(2*length(y_test)))
toc()


p1=plot(abs.(hcat(Parameters[:kernel_params].values...)'),lab="kern_param",yaxis=(:log))
# p2=plot(hcat(Parameters[:kernel_coeffs].values...)',lab="kern_coeffs",yaxis=(:log))
display(plot(p1,layout=1))
# LogArrays = hcat(LogArrays...)
# figure(2); clf();subplot(1,2,1); plot(LogArrays[3,:]); ylabel("Mean(Log likelihood)")
#  subplot(1,2,2); plot(LogArrays[4,:]); ylabel("Median(Log likelihood)")
if doPlot
    figure(2);clf();
    #GIG
    b=model.α; a=1
    mean_GIG = sqrt.(b).*besselk.(1.5,sqrt.(a.*b))./(sqrt.(a).*besselk.(0.5,sqrt.(a.*b)))
    #PG
    mean_PG= 1.0./(2*model.α).*tanh.(model.α/2)
    scatter(X[:,1],X[:,2],c=mean_PG)
    circle = (0:360)/180*pi
    radius = 1.5
    # plot(radius*cos.(circle),radius*sin.(circle),color="k",marker="None",linewidth=1)
    # model.Plotting("logELBO")
    colorbar()
    xlim([-3,3])
    ylim([-3,3])
    figure(3);clf();
    evol_conv = zeros(MaxIter)
    for i in 1:MaxIter
        evol_conv[i] = Convergence(model,i)
    end
    semilogy(evol_conv)
end
