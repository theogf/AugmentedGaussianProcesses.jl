"""Batch Bayesian Support Vector Machine (no inducing points)"""
mutable struct BatchBSVM{T<:Real} <: FullBatchModel{T}
    @commonfields
    @functionfields
    @gaussianparametersfields
    @kernelfields
    α::Vector{T}
    θ::Vector{T}
    """BatchBSVM Constructor"""
    function BatchBSVM(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 100,
                                    kernel=0,AutotuningFrequency::Integer=1,
                                    ϵ::T=T(1e-5),μ_init::Vector{T}=ones(T,1),verbose::Integer=0) where T
        this = new{T}()
        this.ModelType = BSVM;
        this.Name = "Non-Sparse GP Classifier with Bayesian SVM likelihood"
        initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
        initFunctions!(this)
        initKernel!(this,kernel)
        initGaussian!(this,μ_init)
        initLatentVariables!(this)
        this.α = abs.(T.(rand(this.nSamples)))*2;
        this.θ = zero(this.α)
        return this
    end
    """Empty constructor for loading models"""
    function BatchBSVM{T}() where T
        this = new{T}()
        this.ModelType = BSVM
        this.Name = "Full Batch Nonlinear Bayesian SVM"
        initFunctions!(this)
        return this;
    end
end
