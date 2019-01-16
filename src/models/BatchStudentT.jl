"""Batch Student T Gaussian Process Regression (no inducing points)"""
mutable struct BatchStudentT{T<:Real} <: FullBatchModel{T}
    @commonfields
    @functionfields
    @gaussianparametersfields
    @kernelfields
    ν::T
    α::T
    β::Vector{T}
    θ::Vector{T}
    """BatchStudentT Constructor"""
    function BatchStudentT(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),
                                    nEpochs::Integer = 200,
                                    kernel=0,AutotuningFrequency::Integer=1,
                                    ϵ::Real=1e-5,μ_init::Vector{T}=ones(T,1),verbose::Integer=0,ν::T=5.0) where {T<:Real}
            this = new{T}()
            this.ModelType = StudentT
            this.Name = "Non Sparse GP Regression with Student-T Likelihood"
            initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            initGaussian!(this,μ_init);
            this.ν = ν
            this.α = (this.ν+1.0)/2.0
            this.β = abs.(rand(this.nSamples))*2;
            this.θ = zero(this.β)
            return this;
    end
    """Empty constructor for loading models"""
    function BatchStudentT{T}() where T
        this = new{T}()
        this.ModelType = StudentT
        this.Name = "Student T Gaussian Process Regression"
        initFunctions!(this)
        return this;
    end
end
