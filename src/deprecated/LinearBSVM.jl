
#Linear Bayesian Support Vector Machines

mutable struct LinearBSVM{T<:Real} <: LinearModel{T}
    @commonfields
    @functionfields
    @linearfields
    @latentfields
    @gaussianparametersfields

    @stochasticfields
    #Constructor
    function LinearBSVM(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,Autotuning::Bool=false,
                                    nEpochs::Integer = 2000,batchsize::Integer=-1,κ_s::T=one(T),τ_s::Integer=100,
                                    AutotuningFrequency::Integer=4, Intercept::Bool=true,ϵ::T=T(1e-5),μ_init::Vector{T}=ones(T,1),
                                    SmoothingWindow::Integer=5, verbose::Integer=0) where T
        this = new{T}()
        this.ModelType = BSVM
        this.Name = "Linear Bayesian SVM"
        initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency);
        initFunctions!(this)
        initLinear!(this,Intercept);
        initGaussian!(this,μ_init);
        initLatentVariables!(this);
        if Stochastic
            initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
        else
            this.MBIndices = 1:model.nSamples; this.nSamplesUsed = this.nSamples;this.StochCoeff=1.0;
        end
        return this
    end
end
