module LinResp

using DAM
using DataAccess
using Distributions
export GetLastState
export correctedpredict
export meancorrection
export VarCorrection


function GetLastState(dataset;cluster=false)
    top_fold = "../$(cluster?"cluster_":"")results/$(dataset)_SavedParams/SXGPC/"
    models = Array{AugmentedModel,1}(10)
    (X_data,y_data,dataset) = get_Dataset(dataset)
    fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
    for i in 1:10
        X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
        y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
        X_test = readdlm(top_fold*"X_test_$i")
        y_test = readdlm(top_fold*"y_test_$i")
        c = readdlm(top_fold*"c_$i")
        μ = readdlm(top_fold*"mu_$i")
        m = length(μ)
        Σ = readdlm(top_fold*"sigma_$i")
        kernel_coeff = readdlm(top_fold*"kernel_coeff_$i")
        kernel_param = readdlm(top_fold*"kernel_param_$i")
        kerns = [Kernel("rbf",kernel_param,param=kernel_coeff)]
        models[i] = SparseXGPC(X,y;optimizer=Adam(α=0.5),OptimizeIndPoints=true,Stochastic=true,ϵ=1e-4,nEpochs=MaxIter,SmoothingWindow=10,Kernels=kerns,Autotuning=true,AutotuningFrequency=2,VerboseLevel=2,AdaptiveLearningRate=true,BatchSize=BatchSize,m=Ninducingpoints)
        models[i].μ = μ
        models[i].ζ = Σ
        models[i].m = m
        models[i].α = c
    end
    return models
end


function correctedpredict(model,X_test)
    n = size(X_test,1)
    ksize = model.nSamples
    if model.DownMatrixForPrediction == 0
        if model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.invK*model.μ
        end
      model.DownMatrixForPrediction = -(model.invK*(eye(ksize)-model.ζ*model.invK))
    end
    kstar = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
    A = kstar*model.invK
    kstarstar = CreateDiagonalKernelMatrix(X_test,model.Kernel_function)
    meanfstar = kstar*model.TopMatrixForPrediction
    meanfstar_corr = meanfstar + meancorrection(meanfstar,kstar,model.invK,model.μ,model.Σ,model.α)
    covfstar = kstarstar + diag(A*(model.ζ*model.invK-eye(model.nSamples))*transpose(kstar))
    covfstar_corr = covfstar + varcorrection(meanfstar,kstar,model.invK,model.μ,model.Σ,model.α)
    predic = zeros(nPoints)
    for i in 1:nPoints
        d= Normal(m[i],cov[i])
        f=function(x)
            return logit(x)*pdf(d,x)
        end
        predic[i] = quadgk(f,m[i]-10*cov[i],m[i]+10*cov[i])[1]
    end
    return predic,meanfstar_corr,covfstar_corr
end

function meancorrection(fstar,kstar,invK,μ,Σ,c)
    var_c = (tanh.(c.*0.5).^2-1.0)./(4.*c.^2)+(tanh.(c.*0.5))./(2.*c.^3)
    correction = kstar*invK*Σ*(diag(Σ).*μ.*var_c)
    return correction
end

function varcorrection(fstar,kstar,invK,μ,Σ,c)
    return 0
end
end
