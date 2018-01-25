# if !isdefined(:DataAccess); include("DataAccess.jl"); end;
# if !isdefined(:PolyaGammaGPC); include("../src/XGPC.jl"); end;
# if !isdefined(:KernelFunctions); include("KernelFunctions.jl"); end;
 # include("../src/XGPC.jl");
# include("../src/DataAugmentedClassifiers.jl")
# include("../src/DataAugmentedClassifierFunctions.jl")
push!(LOAD_PATH,"/home/theo/XGPC/src/")
using Plots
pyplot()
#unicodeplots()
using DataAccess
using KernelFunctions
using Distributions
include("PGSampler.jl")
include("DataAugmentedModels.jl")
import DAM
doPlot = false
folding = false
X = []; y = []; X_test = []; y_test = [];i=4
# (X_data,y_data,DatasetName) = get_Dataset("BreastCancer")
N_data = 100
N_test = 100
noise = 0.5
function latent(x)
    return sin.(2*x)
end
X_data = sort(rand(N_data))*10.0
X_test = collect(linspace(0,10.0,N_test))
y_data = sign.(1.0./(1.0+exp.(-latent(X_data)+rand(Normal(0,noise),length(X_data))))-0.5)
y_test = sign.(1.0./(1.0+exp.(-latent(X_test)+rand(Normal(0,noise),length(X_test))))-0.5)
X=  X_data; y=y_data
plot(X_data,y_data,t=:scatter,xlim=(0,10),ylim=(-1.5,1.5),lab="Training")
plot!(X_test, y_test,t=:scatter,xlim=(0,10),ylim=(-1.5,1.5),lab="Testing")
plot!(0:0.001:10,latent,lab="latent f")
# data = readdlm("test_file_circle")
# X_data = data[:,1:end-1]; y_data = data[:,end];
# DatasetName = "Test_Circle"
MaxIter = 10000#Maximum number of iterations for every algorithm
if folding
    (nSamples,nFeatures) = size(X_data);
    nFold = 10; #Chose the number of folds
    fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
    #Global variables for debugging
    X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
    y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
    X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
    y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
end
    M=100; θ=1; ϵ=1e-10; γ=1e-3
kerns = [Kernel("rbf",1.0;params=θ)]
# kerns = [Kernel("linear",1.0)]
BatchSize = 100
Ninducingpoints = 100
 tic()
 # model = DAM.BatchXGPC(X,y;Kernels=kerns,Autotuning=true,AutotuningFrequency=2VerboseLevel=0,ϵ=1e-10,nEpochs=100)
model = DAM.BatchXGPC(X,y;Kernels=kerns,Autotuning=false,AutotuningFrequency=2,VerboseLevel=0,ϵ=1e-10,nEpochs=100)
gmodel = DAM.GibbsSamplerGPC(X,y;burninsamples=1000,samplefrequency=1,Kernels=kerns,VerboseLevel=0,ϵ=1e-10,nEpochs=20000)
 # model = DAM.SparseXGPC(X,y;Stochastic=true,ϵ=1e-4,nEpochs=MaxIter,SmoothingWindow=10,Kernels=kerns,Autotuning=true,AutotuningFrequency=4,ρ_AT=0.01,VerboseLevel=2,AdaptiveLearningRate=true,BatchSize=BatchSize,m=Ninducingpoints)

gmodel.train()
model.train()
 # model.train(callback=StoreIt,convergence=function (model::AugmentedClassifier,iter)  return LogLikeConvergence(model,iter,X_test,y_test);end)
y_predic_log = gmodel.Predict(X_test)
plot!(X_data,gmodel.μ,lab="latent f gibbs")
plot!(X_data,model.μ,lab="latent f vi")
println(1-sum(1-y_test.*y_predic_log)/(2*length(y_test)))
toc()

evol_f = hcat(gmodel.samplehistory[:f].values...)
function compute_on_window(fs,window)
    means = zeros(size(fs,2)-window,size(fs,1))
    covs = zeros(size(fs,2)-window,size(fs,1),size(fs,1))
    for i in 1:(size(fs,2)-window)
        means[i,:] = squeeze(mean(fs[:,i:i+(window-1)],2),2)
        covs[i,:,:] = cov(fs[:,i:i+(window-1)],2)
    end
    return means,covs
end
means,covs = compute_on_window(evol_f,1000)
plot(mean(means,2))
plot(covs[:,1,:],lab="")
evol_ω = hcat(gmodel.samplehistory[:ω].values...)
p1 = plot(abs.(model.ζ-gmodel.ζ),t=:heatmap,yaxis=(:flip))
p2 = plot(gmodel.ζ,t=:heatmap,yaxis=(:flip))
p3 = plot(model.ζ,t=:heatmap,yaxis=(:flip))
plot(p1,p2,p3,layout=3)
if doPlot
    p1 = plot(mean(evol_f,1)',lab="Mean f")
    p2 = plot(var(evol_f,1)',lab="Var f")
    p3 = plot(mean(evol_ω,1)',lab="Mean ω")
    p4 = plot(var(evol_ω,1)',lab="Var ω")
    display(plot(p1,p2,p3,p4,layout=(2,2)))
    savefig("../plots/VariablesEvolution.png")
    p1 = plot([model.μ gmodel.μ],lab=["VI" "Gibbs"],xlabel="dim",ylabel="f_i",grid=:off)
    p2 = plot(abs.(model.μ-gmodel.μ),lab="|f_VI-f_Gibbs|",xlabel="dim",grid=:off)
    display(plot(p1,p2,layout=2))
    savefig("../plots/Error_f.png")
end
