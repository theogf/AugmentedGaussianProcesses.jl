"""Classic Batch Gaussian Process Regression (no inducing points)"""
mutable struct BatchGPRegression{T<:Real} <: FullBatchModel{T}
    @commonfields
    @functionfields
    @kernelfields
    gnoise::T
    """Constructor for the full batch GP Regression"""
    function BatchGPRegression(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=true,nEpochs::Integer = 10,
                                    kernel=0,noise::T=1e-3,verbose::Integer=0) where {T<:Real}
            this = new{T}()
            this.ModelType = Regression
            this.Name = "Non Sparse GP Regression with Gaussian Likelihood"
            initCommon!(this,X,y,1e-16,nEpochs,verbose,Autotuning,1,Adam());
            initFunctions!(this);
            initKernel!(this,kernel);
            this.gnoise = noise
            return this;
    end
end
