module PerturbativeCorrection

# using Distributions
# using KernelFunctions
export GetLastState
export correctedpredict
export meancorrection
export VarCorrection


# function GetLastState(dataset;cluster=false)
#     top_fold = "../$(cluster?"cluster_":"")results/$(dataset)_SavedParams/SXGPC/"
#     models = Array{AugmentedModel,1}(10)
#     (X_data,y_data,dataset) = get_Dataset(dataset)
#     fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
#     for i in 1:10
#         X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
#         y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
#         X_test = readdlm(top_fold*"X_test_$i")
#         y_test = readdlm(top_fold*"y_test_$i")
#         c = readdlm(top_fold*"c_$i")
#         μ = readdlm(top_fold*"mu_$i")
#         m = length(μ)
#         Σ = readdlm(top_fold*"sigma_$i")
#         kernel_coeff = readdlm(top_fold*"kernel_coeff_$i")
#         kernel_param = readdlm(top_fold*"kernel_param_$i")
#         kerns = [Kernel("rbf",kernel_param,param=kernel_coeff)]
#         models[i] = SparseXGPC(X,y;optimizer=Adam(α=0.5),OptimizeIndPoints=true,Stochastic=true,ϵ=1e-4,nEpochs=MaxIter,SmoothingWindow=10,Kernels=kerns,Autotuning=true,AutotuningFrequency=2,VerboseLevel=2,AdaptiveLearningRate=true,BatchSize=BatchSize,m=Ninducingpoints)
#         models[i].μ = μ
#         models[i].ζ = Σ
#         models[i].m = m
#         models[i].α = c
#     end
#     return models
# end


function logit(x)
    return 1./(1+exp.(-x))
end

function correctedpredict(model,X_test)
    n = size(X_test,1)
    kstar = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
    kstarstar = CreateDiagonalKernelMatrix(X_test,model.Kernel_function)
    A = kstar*model.invK
    meanfstar = A*model.μ
    meanfstar_corr = meanfstar + meancorrection(model.μ,model.ζ,model.α,A*model.ζ)
    covfstar = kstarstar + diag(A*(model.ζ*model.invK-eye(model.nSamples))*transpose(kstar))
    covfstar_corr = covfstar + varcorrection(meanfstar,model.μ,model.ζ,model.α;fstar_corr=meanfstar_corr)
    predic = zeros(n)
    for i in 1:n
        d= Normal(meanfstar_corr[i],covfstar_corr[i])
        f=function(x)
            return logit(x)*pdf(d,x)
        end
        predic[i] = quadgk(f,meanfstar_corr[i]-10*covfstar_corr[i],meanfstar_corr[i]+10*covfstar_corr[i])[1]
    end
    return predic,meanfstar_corr,covfstar_corr
end

function correctedlatent(model)
    mean_corr = model.μ .+ meancorrection(model.μ,model.ζ,model.α,model.ζ)
    var_corr = diag(model.ζ)
    # correction = model.μ.^2 - mean_corr.^2
    correction = 0.5*(model.ζ).^2*((model.μ.^2+diag(model.ζ)).^2.*var_c)
    var_corr += correction
    return mean_corr,var_corr
end

function meancorrection(μ,Σ,c,covfefe)
    var_c = (tanh.(c.*0.5).^2-1.0)./(4.*c.^2)+(tanh.(c.*0.5))./(2.*c.^3)
    correction = covfefe*(diag(Σ).*μ.*var_c)
    return correction
end

function varcorrection(fstar,μ,Σ,c,covfefe;fstar_corr=0)
    var_c = (tanh.(c.*0.5).^2-1.0)./(4.*c.^2)+(tanh.(c.*0.5))./(2.*c.^3)
    if fstar_corr ==0
        fstar_corr = fstar + meancorrection(μ,Σ,c,covfefe)
    end
    corr_mean = fstar_corr - fstar
    # correction = fstar.^2 - fstar_corr.^2
    correction = 0.5*(covfefe.^2*((μ.^2-diag(Σ)).*var_c))
    # display(Plots.plot(covfstar,correction,t=:scatter))

    println(correction)
    return correction
end

end
