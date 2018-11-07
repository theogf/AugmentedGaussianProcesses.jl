"""Batch Bayesian Support Vector Machine (no inducing points)"""
mutable struct BatchBSVM <: FullBatchModel
    @commonfields
    @functionfields
    @gaussianparametersfields
    @kernelfields
    α::Vector{Float64}
    θ::Vector{Float64}
    """BatchBSVM Constructor"""
    function BatchBSVM(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 100,
                                    kernel=0,noise::Real=1e-3,AutotuningFrequency::Integer=1,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],verbose::Integer=0)
        this = new()
        this.ModelType = BSVM;
        this.Name = "Non-Sparse GP Classifier with Bayesian SVM likelihood"
        initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
        initFunctions!(this)
        initKernel!(this,kernel)
        initGaussian!(this,μ_init)
        initLatentVariables!(this)
        this.α = abs.(rand(this.nSamples))*2;
        this.θ = zero(this.α)
        return this
    end
    """Empty constructor for loading models"""
    function BatchBSVM()
        this = new()
        this.ModelType = BSVM
        this.Name = "Full Batch Nonlinear Bayesian SVM"
        initFunctions!(this)
        return this;
    end
end
