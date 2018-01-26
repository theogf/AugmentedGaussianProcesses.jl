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
doPlot = false
N_data = 100
N_test = 100
noise = 0.01
function latent(x)
    return sin.(2*x)
end
# X_data = sort(rand(N_data))*10.0
X_data = rand(Normal(2.5,0.5),Int64(N_data/2))
X_data = sort(vcat(X_data,rand(Normal(7.5,0.5),Int64(N_data/2))))
X_test = collect(linspace(0,10.0,N_test))
y_data = sign.(1.0./(1.0+exp.(-latent(X_data)+rand(Normal(0,noise),length(X_data))))-0.5)
y_test = sign.(1.0./(1.0+exp.(-latent(X_test)+rand(Normal(0,noise),length(X_test))))-0.5)
X=  X_data; y=y_data
iter_points = collect(1:1:1000)

MaxIter = 10000#Maximum number of iterations for every algorithm
metrics = MVHistory()
Parameters = MVHistory()
function StoreIt(model::DAM.AugmentedModel,iter)#;iter_points=[],LogArrays=[],X_test=0,y_test=0)
    if in(iter,iter_points)
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
        # println("Iteration $iter : Accuracy is $(1-sum(1-y_test.*sign.(y_p-0.5))/(2*length(y_test))), ELBO is $(DAM.ELBO(model)), θ is $(model.Kernels[1].param)")
        push!(Parameters,:μ,iter,model.μ)
        push!(Parameters,:diag_ζ,iter,diag(model.ζ))
        push!(Parameters,:kernel_params,iter,getfield.(model.Kernels,:param))
        push!(Parameters,:kernel_coeffs,iter,getfield.(model.Kernels,:coeff))
    end
end
θ=1; ϵ=1e-10; γ=0.1
nBurnin = 100; nSamples = 100000+nBurnin;
kerns = [Kernel("rbf",1.0;params=θ)]
model = DAM.BatchXGPC(X,y;Kernels=kerns,Autotuning=false,VerboseLevel=0,ϵ=1e-10,nEpochs=100)
gmodel = DAM.GibbsSamplerGPC(X,y;burninsamples=nBurnin,samplefrequency=1,Kernels=kerns,VerboseLevel=0,ϵ=1e-10,nEpochs=nSamples)

gmodel.train()
model.train(callback=StoreIt)
 # model.train(callback=StoreIt,convergence=function (model::AugmentedClassifier,iter)  return LogLikeConvergence(model,iter,X_test,y_test);end)
y_predic_log = gmodel.Predict(X_test)
plot(X_data,y_data,t=:scatter,xlim=(0,10),ylim=(-3,3),lab="Training")
# plot!(X_test, y_test,t=:scatter,xlim=(0,10),ylim=(-1.5,1.5),lab="Testing")
X_range = collect(0:00.1:10)
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
