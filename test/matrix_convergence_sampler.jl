push!(LOAD_PATH,"/home/theo/XGPC/src/")
using Plots
pyplot()
#unicodeplots()
include("/home/theo/XGPC/src/DataAugmentedModels.jl")
using DataAccess
using KernelFunctions
using ValueHistories
using Distributions
using PGSampler
import DAM

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

iter_points = collect(1:1:1000)
θ=1.0; ϵ=1e-10; γ=0.001
nBurnin = 100; nSamples = 100000+nBurnin;
kerns = [Kernel("rbf",1.0;params=θ)]
model = DAM.BatchXGPC(X,y;Kernels=kerns,Autotuning=false,VerboseLevel=0,ϵ=1e-10,nEpochs=100)
gmodel = DAM.GibbsSamplerGPC(X,y;burninsamples=nBurnin,samplefrequency=1,Kernels=kerns,VerboseLevel=0,ϵ=1e-10,nEpochs=nSamples)

MaxIter = 10000 #Maximum number of iterations for every algorithm

gmodel.train()
model.train()

y_predic_log = gmodel.Predict(X_test)
plot(X_data,y_data,t=:scatter,xlim=(0,10),ylim=(-3,3),lab="Training")
fstar_vi,covfstar_vi = DAM.computefstar(model,X_range)
fstar_gibbs,covfstar_gibbs = DAM.computefstar(gmodel,X_range)
plot!(X_range,latent,lab="latent f")
# plot!(X_data,gmodel.μ,lab="latent f gibbs",color=:red)
# plot!(X_data,gmodel.μ+2*sqrt.(diag(gmodel.ζ)),l=1,lab="",color=:red,fillalpha=0.1,fillrange=gmodel.μ-2*sqrt.(diag(gmodel.ζ)))
# plot!(X_data,gmodel.μ-2*sqrt.(diag(gmodel.ζ)),l=1,lab="",color=:red)
# plot!(X_data,model.μ,lab="latent f vi",color=:blue)
# plot!(X_data,model.μ-2*sqrt.(diag(model.ζ)),l=1,lab="",color=:blue)
# display(plot!(X_data,model.μ+2*sqrt.(diag(model.ζ)),l=1,lab="",color=:blue,fillalpha=0.1,fillrange=model.μ-2*sqrt.(diag(model.ζ))))
plot!(X_range,fstar_gibbs,lab="latent f gibbs",color=:red)
plot!(X_range,fstar_gibbs+2*sqrt.(covfstar_gibbs),l=1,lab="",color=:red,fillalpha=0.1,fillrange=fstar_gibbs-2*sqrt.(covfstar_gibbs))
plot!(X_range,fstar_gibbs-2*sqrt.(covfstar_gibbs),l=1,lab="",color=:red)
plot!(X_range,fstar_vi,lab="latent f vi",color=:blue)
plot!(X_range,fstar_vi-2*sqrt.(covfstar_vi),l=1,lab="",color=:blue)
display(plot!(X_range,fstar_vi+2*sqrt.(covfstar_vi),l=1,lab="",color=:blue,fillalpha=0.1,fillrange=fstar_vi-2*sqrt.(covfstar_vi)))
sleep(5)
println("Accuracy : $(1-sum(1-y_test.*y_predic_log)/(2*length(y_test)))")

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
# means,covs = compute_on_window(evol_f,1000)
# plot(mean(means,2))
# plot(covs[:,1,:],lab="")
# evol_ω = hcat(gmodel.samplehistory[:ω].values...)
# p1 = plot(abs.(model.ζ-gmodel.ζ),t=:heatmap,yaxis=(:flip))
# p2 = plot(gmodel.ζ,t=:heatmap,yaxis=(:flip))
# p3 = plot(model.ζ,t=:heatmap,yaxis=(:flip))
# plot(p1,p2,p3,layout=3)
# if doPlot
#     p1 = plot(mean(evol_f,1)',lab="Mean f")
#     p2 = plot(var(evol_f,1)',lab="Var f")
#     p3 = plot(mean(evol_ω,1)',lab="Mean ω")
#     p4 = plot(var(evol_ω,1)',lab="Var ω")
#     display(plot(p1,p2,p3,p4,layout=(2,2)))
#     savefig("../plots/VariablesEvolution.png")
#     p1 = plot([model.μ gmodel.μ],lab=["VI" "Gibbs"],xlabel="dim",ylabel="f_i",grid=:off)
#     p2 = plot(abs.(model.μ-gmodel.μ),lab="|f_VI-f_Gibbs|",xlabel="dim",grid=:off)
#     display(plot(p1,p2,layout=2))
#     savefig("../plots/Error_f.png")
# end
#
# niter = length(Parameters[:μ].values)
# mus = Parameters[:μ].values
# zetas = Parameters[:diag_ζ].values
#
# anim = @animate for i in 1:niter
#     plot(X_data,y_data,t=:scatter,xlim=(0,10),ylim=(-2.5,2.5),lab="Training")
#     plot!(X_test, y_test,t=:scatter,xlim=(0,10),ylim=(-2.5,2.5),lab="Testing")
#     plot!(0:0.001:10,latent,lab="latent f")
#     plot!(X_data,gmodel.μ,lab="latent f gibbs",color=:red)
#     plot!(X_data,gmodel.μ+2*sqrt.(diag(gmodel.ζ)),l=1,lab="",color=:red,fillalpha=0.1,fillrange=gmodel.μ-2*sqrt.(diag(gmodel.ζ)))
#     plot!(X_data,gmodel.μ-2*sqrt.(diag(gmodel.ζ)),l=1,lab="",color=:red)
#     plot!(X_data,mus[i],lab="latent f vi",color=:blue)
#     plot!(X_data,mus[i]-2*sqrt.(zetas[i]),l=1,lab="",color=:blue)
#     plot!(X_data,mus[i]+2*sqrt.(zetas[i]),l=1,lab="",color=:blue,fillalpha=0.1,fillrange=mus[i]-2*sqrt.(zetas[i]))
# end every 1
#
# gif(anim,"convergence.gif")
