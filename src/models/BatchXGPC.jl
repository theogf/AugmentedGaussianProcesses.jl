"""Batch Gaussian Process Classifier with Logistic Likelihood (no inducing points)"""
mutable struct BatchXGPC{T<:Real} <: FullBatchModel{T}
    @commonfields
    @functionfields
    @gaussianparametersfields
    @kernelfields
    c::Vector{T}
    θ::Vector{T}
    """BatchXGPC Constructor"""
    function BatchXGPC(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 200,
                                    kernel=0,AutotuningFrequency::Integer=1,
                                    ϵ::T=1e-5,μ_init::Vector{T}=ones(T,1),verbose::Integer=0) where {T<:Real}
            this = new{T}()
            this.ModelType = XGPC
            this.Name = "Non Sparse GP Classifier with Logistic Likelihood"
            initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            initGaussian!(this,μ_init);
            this.c = abs.(rand(this.nSamples))*2;
            this.θ = zero(this.c)
            return this;
    end
    """Empty constructor for loading models"""
    function BatchXGPC{T}() where T
        this = new{T}()
        this.ModelType = XGPC
        this.Name = "Gaussian Process Classifier with Logistic Likelihood"
        initFunctions!(this)
        return this;
    end
end
