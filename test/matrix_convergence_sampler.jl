push!(LOAD_PATH,"/home/theo/XGPC/src/")
using Plots
pyplot()
#unicodeplots()
include("/home/theo/XGPC/src/DataAugmentedModels.jl")
using DataAccess
using PyCall
using KernelFunctions
using ValueHistories
using Distributions
using PGSampler
import DAM

@pyimport gpflow
(X_data,y_data,DatasetName) = get_Dataset("German")
(nSamples,nFeatures) = size(X_data);

nFold = 3; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
#Global variables for debugging
X = []; y = []; X_test = []; y_test = [];i=1
X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]

var=100.0; θ=5.0; ϵ=1e-10; γ=0.001
nBurnin = 100; nSamples = 10000+nBurnin;
MaxIter = 5000 #Maximum number of iterations for every algorithm

kerns = [Kernel("rbf",var;params=θ)]
vimodel = DAM.BatchXGPC(X,y;Kernels=kerns,Autotuning=false,VerboseLevel=1,ϵ=1e-10,nEpochs=100)
vgpcmodel = gpflow.vgp[:VGP](X, reshape((y+1)./2,(size(y,1),1)),kern=gpflow.kernels[:Add]([gpflow.kernels[:RBF](nFeatures,variance=var,lengthscales=θ,ARD=false),gpflow.kernels[:White](input_dim=nFeatures,variance=γ)]), likelihood=gpflow.likelihoods[:Bernoulli](invlink=gpflow.likelihoods[:logit]))
vgpcmodel[:kern][:fixed] = true
gibbsmodel = DAM.GibbsSamplerGPC(X,y;burninsamples=nBurnin,samplefrequency=1,Kernels=kerns,VerboseLevel=1,ϵ=1e-10,nEpochs=nSamples)


gibbsmodel.train()
y_gibbs = gibbsmodel.predict(X_test)

vimodel.train(iterations=MaxIter)
y_vi = vimodel.predict(X_test)

vgpcmodel[:optimize](maxiter=MaxIter)
y_vgpc = sign.(vgpcmodel[:predict_y](X_test)[1][:]*2-1)

println("Accuracy : Gibbs $(1-sum(1-y_test.*y_gibbs)/(2*length(y_test)))
                    VGPC  $(1-sum(1-y_test.*y_vgpc)/(2*length(y_test)))
                    XGPC  $(1-sum(1-y_test.*y_vi)/(2*length(y_test)))")


# evol_f = hcat(gmodel.samplehistory[:f].values...)
# function compute_on_window(fs,window)
#     means = zeros(size(fs,2)-window,size(fs,1))
#     covs = zeros(size(fs,2)-window,size(fs,1),size(fs,1))
#     for i in 1:(size(fs,2)-window)
#         means[i,:] = squeeze(mean(fs[:,i:i+(window-1)],2),2)
#         covs[i,:,:] = cov(fs[:,i:i+(window-1)],2)
#     end
#     return means,covs
# end


function save_values(dataset,gibbs_m,vgpc_m,vi_m,X_test,y_test)
    p_gibbs = gibbs_m.predictproba(X_test)
    f_gibbs,covf_gibbs = DAM.computefstar(gibbs_m,X_test)
    p_vgpc, = vgpc_m[:predict_y](X_test)
    f_vgpc, covf_vgpc = vgpc_m[:predict_f](X_test)
    p_vi = vi_m.predictproba(X_test)
    f_vi, covf_vi = DAM.computefstar(vi_m,X_test)
    top_fold = "/home/theo/XGPC/results/"*dataset
    if !isdir(top_fold); mkdir(top_fold); end;
    data_gibbs = hcat(p_gibbs,f_gibbs,covf_gibbs)
    writedlm(top_fold*"/Gibbs",data_gibbs)
    data_vgpc = hcat(p_vgpc,f_vgpc,covf_vgpc)
    writedlm(top_fold*"/VGPC",data_vgpc)
    data_vi = hcat(p_vi,f_vi,covf_vi)
    writedlm(top_fold*"/XGPC",data_vi)
end

save_values(DatasetName,gibbsmodel,vgpcmodel,vimodel,X_test,y_test)



function truth_comparison(dataset)
    truth = readdlm("/home/theo/XGPC/results/"*dataset*"/Gibbs")
    p_truth = truth[:,1]; f_truth = truth[:,2]; covf_truth = truth[:,3]
    xgpc = readdlm("/home/theo/XGPC/results/"*dataset*"/XGPC")
    p_xgpc = xgpc[:,1]; f_xgpc = xgpc[:,2]; covf_xgpc = xgpc[:,3]
    vgpc = readdlm("/home/theo/XGPC/results/"*dataset*"/VGPC")
    p_vgpc = vgpc[:,1]; f_vgpc = vgpc[:,2]; covf_vgpc = vgpc[:,3]
    p1 = plot(p_truth,p_xgpc,t=:scatter, lab="p XGPC", xlim=(0,1),ylim=(0,1))
    p2 = plot(f_truth,f_xgpc,t=:scatter, lab="μ XGPC",title="θ=$(θ), var=$(var)")
    p3 = plot(covf_truth,covf_xgpc,t=:scatter, lab="σ XGPC")
    p4 = plot(p_truth,p_vgpc,t=:scatter,lab="p VGPC", xlim=(0,1),ylim=(0,1))
    p5 = plot(p_truth,f_vgpc,t=:scatter,lab="μ VGPC")
    p6 = plot(p_truth,covf_vgpc,t=:scatter,lab="σ VGPC")
    plot(p1,p2,p3,p4,p5,p6,layout=(2,3))
end

p = truth_comparison("German")
display(p)
savefig(p,"/home/theo/XGPC/plots/ModelApproximation/"*DatasetName*"/theta_$(θ)_var_$(var).png")
