push!(LOAD_PATH,"/home/theo/XGPC/src/")
using Plots
pyplot()
include("/home/theo/XGPC/src/DataAugmentedModels.jl")
include("/home/theo/XGPC/src/correction.jl")
using DataAccess
using PyCall
using KernelFunctions
using ValueHistories
using Distributions
using PGSampler
import OMGP
import LinResp

@pyimport gpflow
(X_data,y_data,DatasetName) = get_Dataset("Diabetis")
(nSamples,nFeatures) = size(X_data);

nFold = 4; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
#Global variables for debugging
X = []; y = []; X_test = []; y_test = [];i=1
X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]

k_weight=100.0; θ=5.0; ϵ=1e-10; γ=0.001
nBurnin = 100; nSamples = 10000+nBurnin;
MaxIter = 5000 #Maximum number of iterations for every algorithm

kerns = [Kernel("rbf",k_weight;params=θ)]
vimodel = DAM.BatchXGPC(X,y;Kernels=kerns,Autotuning=false,VerboseLevel=1,ϵ=1e-10,nEpochs=1000)
vgpcmodel = gpflow.vgp[:VGP](X, reshape((y+1)./2,(size(y,1),1)),kern=gpflow.kernels[:Add]([gpflow.kernels[:RBF](nFeatures,variance=k_weight,lengthscales=θ,ARD=false),gpflow.kernels[:White](input_dim=nFeatures,variance=γ)]), likelihood=gpflow.likelihoods[:Bernoulli](invlink=gpflow.likelihoods[:logit]))
vgpcmodel[:kern][:fixed] = true
gibbsmodel = DAM.GibbsSamplerGPC(X,y;burninsamples=nBurnin,samplefrequency=1,Kernels=kerns,VerboseLevel=2,ϵ=1e-10,nEpochs=nSamples)

println("Started sampling")
gibbsmodel.train()
y_gibbs = gibbsmodel.predict(X_test)

println("Started XGPC")
vimodel.train(iterations=MaxIter)
y_vi = vimodel.predict(X_test)

println("Started VGPC")
vgpcmodel[:optimize](maxiter=MaxIter)
y_vgpc = sign.(vgpcmodel[:predict_y](X_test)[1][:]*2-1)

println("Accuracy : Gibbs $(1-sum(1-y_test.*y_gibbs)/(2*length(y_test)))
                    VGPC  $(1-sum(1-y_test.*y_vgpc)/(2*length(y_test)))
                    XGPC  $(1-sum(1-y_test.*y_vi)/(2*length(y_test)))")

function save_values(dataset,X_test,y_test,gibbs_m=0,vgpc_m=0,vi_m=0)
    top_fold = "/home/theo/XGPC/results/ComparisonExp"*dataset
    if !isdir(top_fold); mkdir(top_fold); end;
    if vi_m != 0
        p_vi = vi_m.predictproba(X_test)
        f_vi, covf_vi = DAM.computefstar(vi_m,X_test)
        p_vi_corr, f_vi_corr, covf_vi_corr = LinResp.correctedpredict(vi_m,X_test)
        data_vi = hcat(p_vi,f_vi,covf_vi)
        data_vi_latent = hcat(vi_m.μ,diag(vi_m.ζ))
        writedlm(top_fold*"/XGPC",data_vi)
        writedlm(top_fold*"/XGPC_latent",data_vi_latent)
        data_vi_corr = hcat(p_vi_corr,f_vi_corr,covf_vi_corr)
        f_latent, covf_latent = LinResp.correctedlatent(vi_m)
        data_vi_latent_corr = hcat(f_latent,covf_latent)
        writedlm(top_fold*"/CorrXGPC",data_vi_corr)
        writedlm(top_fold*"/CorrXGPC_latent",data_vi_latent_corr)
    end
    if gibbs_m != 0
        p_gibbs = gibbs_m.predictproba(X_test)
        f_gibbs,covf_gibbs = DAM.computefstar(gibbs_m,X_test)
        data_gibbs = hcat(p_gibbs,f_gibbs,covf_gibbs)
        data_gibbs_latent =  hcat(gibbs_m.μ,diag(gibbs_m.ζ))
        writedlm(top_fold*"/Gibbs",data_gibbs)
        writedlm(top_fold*"/Gibbs_latent",data_gibbs_latent)
    end
    if vgpc_m != 0
        p_vgpc, = vgpc_m[:predict_y](X_test)
        f_vgpc, covf_vgpc = vgpc_m[:predict_f](X_test)
        data_vgpc = hcat(p_vgpc,f_vgpc,covf_vgpc)
        writedlm(top_fold*"/VGPC",data_vgpc)
    end
end

function estimate_improvement(gibbs_model,vi_model,X_test,y_test)
    p_vi = vi_model.predictproba(X_test)
    f_vi, covf_vi = DAM.computefstar(vi_model,X_test)
    y_vi = deepcopy(p_vi); y_vi = sign.(y_vi-0.5)
    ll_vi = zeros(y_test); ll_vi[y_test.==1] = log.(p_vi[y_test.==1]); ll_vi[y_test.==-1] = log.(1-p_vi[y_test.==-1])
    acc_vi = 1-sum(1-y_vi.*y_test)/(2*length(y_test))
    p_corr, f_corr, covf_corr = LinResp.correctedpredict(vi_model,X_test)
    y_corr = deepcopy(p_corr); y_corr = sign.(y_corr-0.5)
    ll_corr = zeros(y_test); ll_corr[y_test.==1] = log.(p_corr[y_test.==1]); ll_corr[y_test.==-1] = log.(1-p_corr[y_test.==-1])
    acc_corr = 1-sum(1-y_corr.*y_test)/(2*length(y_test))
    p_gibbs = gibbs_model.predictproba(X_test)
    f_gibbs,covf_gibbs = DAM.computefstar(gibbs_model,X_test)
    ll_gibbs = zeros(y_test); ll_gibbs[y_test.==1] = log.(p_gibbs[y_test.==1]); ll_gibbs[y_test.==-1] = log.(1-p_gibbs[y_test.==-1])
    acc_gibbs = 1-sum(1-y_test.*y_gibbs)/(2*length(y_test))
    return (p_vi,f_vi,covf_vi,-ll_vi,acc_vi),(p_corr,f_corr,covf_corr,-ll_corr,acc_corr),(p_gibbs,f_gibbs,covf_gibbs,-ll_gibbs,acc_gibbs)
end


function compute_improvement(vi,corr,truth)
    d_mu_vi = mean(abs.(vi[2]-truth[2])); d_mu_corr = mean(abs.(corr[2]-truth[2]))
    d_sig_vi = mean(abs.(vi[3]-truth[3])); d_sig_corr = mean(abs.(corr[3]-truth[3]))
    acc_vi = vi[5]; acc_corr = corr[5]
    ll_vi = mean(abs.(vi[4]-truth[4])); ll_corr = mean(abs.(corr[4]-truth[4]))
    println("diff mu : $(d_mu_vi) / $(d_mu_corr)")
    println("diff sig : $(d_sig_vi) / $(d_sig_corr)")
    println("acc : $(acc_vi) / $(acc_corr) / $(truth[5])")
    println("diff loglike : $(ll_vi) / $(ll_corr)")
    println("mean ll : $(mean(vi[4])) / $(mean(corr[4])) / $(mean(truth[4]))")
end
compute_improvement(estimate_improvement(gibbsmodel,vimodel,X_test,y_test)...)

println("Bam")



function truth_predict_comparison(dataset;improvement=false)
    truth = readdlm("/home/theo/XGPC/results/ComparisonExp"*dataset*"/Gibbs")
    p_truth = truth[:,1]; f_truth = truth[:,2]; covf_truth = truth[:,3]
    xgpc = readdlm("/home/theo/XGPC/results/ComparisonExp"*dataset*"/XGPC")
    p_xgpc = xgpc[:,1]; f_xgpc = xgpc[:,2]; covf_xgpc = xgpc[:,3]
    xgpc_corr = readdlm("/home/theo/XGPC/results/ComparisonExp"*dataset*"/CorrXGPC")
    p_xgpc_corr = xgpc_corr[:,1]; f_xgpc_corr = xgpc_corr[:,2]; covf_xgpc_corr = xgpc_corr[:,3]
    vgpc = readdlm("/home/theo/XGPC/results/ComparisonExp"*dataset*"/VGPC")
    p_vgpc = vgpc[:,1]; f_vgpc = vgpc[:,2]; covf_vgpc = vgpc[:,3]
    offsetmu = 0.2
    offsetsig = 1.0
    offsetp = 0.01
    msize = 3.0
    alphafill = 0.6
    tickssize = 50.0
    p1 = plot(p_truth,p_xgpc,t=:scatter, title="p", xlim=(0,1),ylim=(0,1),lab="",color=:black,markersize=msize,grid=false)
    p1 = plot!(p1,x->x,alpha=0.8,linewidth=msize,lab="")
    p1 = plot!(p1,p_truth,p_xgpc,t=:scatter, title="p", xlim=(0,1),ylim=(0,1),lab="",color=:black,markersize=msize,xtickfont=font(50.0,"sans-serif"),ytickfont=font(25.0,"sans-serif"),titlefont=font(36.0,"sans-serif"))
    p2 = plot(f_truth,f_xgpc,t=:scatter, title="μ",lab="",color=:black,markersize=msize)
    p2 = plot!(p2,x->x,alpha=0.8,linewidth=msize,lab="",grid=false)
    p2 = plot!(p2,f_truth,f_xgpc,t=:scatter, title="μ",lab="",color=:black,markersize=msize,xtickfont=font(25.0,"sans-serif"),ytickfont=font(25.0,"sans-serif"),titlefont=font(36.0,"sans-serif"))
    p3 = plot(covf_truth,covf_xgpc,t=:scatter, title="σ",lab="",color=:black,markersize=msize)
    p3 = plot!(p3,x->x,alpha=0.8,linewidth=msize,lab="",grid=false)
    p3 = plot!(p3,covf_truth,covf_xgpc,t=:scatter, title="σ",lab="",color=:black,markersize=msize,xtickfont=font(25.0,"sans-serif"),ytickfont=font(25.0,"sans-serif"),titlefont=font(36.0,"sans-serif"))
    p4 = plot(p_truth,p_xgpc_corr,t=:scatter, title="p corrected", xlim=(0,1),ylim=(0,1),lab="",color=:black,markersize=msize)
    p4 = plot!(p4,x->x,alpha=0.8,linewidth=msize,lab="",grid=false)
    p4 = plot!(p4,p_truth,p_xgpc_corr,t=:scatter, title="p corrected", xlim=(0,1),ylim=(0,1),lab="",color=:black,markersize=msize,xtickfont=font(25.0,"sans-serif"),ytickfont=font(25.0,"sans-serif"),titlefont=font(36.0,"sans-serif"))
    p5 = plot(f_truth,f_xgpc_corr,t=:scatter, title="μ corrected ",lab="",color=:black,markersize=msize)
    p5 = plot!(p5,x->x,alpha=0.8,linewidth=msize,lab="",grid=false)
    p5 = plot!(p5,f_truth,f_xgpc_corr,t=:scatter, title="μ corrected",lab="",color=:black,markersize=msize,xtickfont=font(25.0,"sans-serif"),ytickfont=font(25.0,"sans-serif"),titlefont=font(36.0,"sans-serif"))
    p6 = plot(covf_truth,covf_xgpc_corr,t=:scatter, title="σ corrected",lab="",color=:black,markersize=msize)
    p6 = plot!(p6,x->x,alpha=0.8,linewidth=msize,lab="",grid=false)
    p6 = plot!(p6,covf_truth,covf_xgpc_corr,t=:scatter, title="σ corrected",lab="",color=:black,markersize=msize,xtickfont=font(25.0,"sans-serif"),ytickfont=font(25.0,"sans-serif"),titlefont=font(36.0,"sans-serif"))
    if improvement
        p7 = plot(p_truth,abs.(p_truth-p_xgpc)-abs.(p_truth-p_xgpc_corr),t=:scatter,zcolor=f_truth,lab="p improvement", xlim=(0,1))
        p7 = plot!(p7,x->0,lab="")
        p8 = plot(f_truth,abs.(f_xgpc - f_truth)-abs.(f_xgpc_corr-f_truth),t=:scatter,lab="μ improvement")
        p8 = plot!(p8,x->0,lab="")
        p9 = plot(covf_truth,abs.(covf_xgpc-covf_truth)-abs.(covf_xgpc_corr-covf_truth),t=:scatter,lab="σ improvement")
        p9 = plot!(p9,x->0,lab="")
        plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,layout=(3,3))
    else
        p = plot(p2,p5,p3,p6,p1,p4,layout=(3,2),size=(600,900))#,right_margin=25*Plots.px)
    end
end
p = truth_predict_comparison(DatasetName,improvement=false)
gui(p)
savefig(p,"../plots/approximation.eps")
println("yuhu")
function truth_latent_comparison(dataset)
    truth = readdlm("/home/theo/XGPC/results/ComparisonExp"*dataset*"/Gibbs_latent")
    f_truth = truth[:,1]; covf_truth = truth[:,2]
    xgpc = readdlm("/home/theo/XGPC/results/ComparisonExp"*dataset*"/XGPC_latent")
    f_xgpc = xgpc[:,1]; covf_xgpc = xgpc[:,2]
    xgpc_corr = readdlm("/home/theo/XGPC/results/ComparisonExp"*dataset*"/CorrXGPC_latent")
    f_xgpc_corr = xgpc_corr[:,1]; covf_xgpc_corr = xgpc_corr[:,2]
    p2 = plot(f_truth,f_xgpc,t=:scatter, lab="μ XGPC",title="$dataset : θ=$(θ), var=$(var)")
    p2 = plot!(p2,x->x,lab="")
    p3 = plot(covf_truth,covf_xgpc,t=:scatter, lab="σ XGPC", zcolor=f_xgpc)
    p3 = plot!(p3,x->x,lab="")
    p5 = plot(f_truth,f_xgpc_corr,t=:scatter, lab="μ corr. XGPC")
    p5 = plot!(p5,x->x,lab="")
    p6 = plot(covf_truth,covf_xgpc_corr,t=:scatter, zcolor=f_xgpc_corr,lab="σ corr. XGPC")
    p6 = plot!(p6,x->x,lab="")
    p8 = plot(f_truth,abs.(f_xgpc - f_truth)-abs.(f_xgpc_corr-f_truth),t=:scatter,lab="μ improvement")
    p8 = plot!(p8,x->0,lab="")
    p9 = plot(covf_truth,abs.(covf_xgpc-covf_truth)-abs.(covf_xgpc_corr-covf_truth),t=:scatter,lab="σ improvement")
    p9 = plot!(p9,x->0,lab="")
    plot(p2,p3,p5,p6,p8,p9,layout=(3,2))
end
save_values(DatasetName,X_test,y_test,gibbsmodel,vgpcmodel,vimodel)

display(p)
p2 = truth_latent_comparison(DatasetName)
display(p2)
# savefig(p,"/home/theo/XGPC/plots/ModelApproximation/"*DatasetName*"/theta_$(θ)_var_$(var).png")
