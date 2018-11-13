"""Classic Batch Gaussian Process Regression (no inducing points)"""
mutable struct BatchGPRegression <: FullBatchModel
    @commonfields
    @functionfields
    @kernelfields
    gnoise::KernelModule.HyperParameter{Float64}
    """Constructor for the full batch GP Regression"""
    function BatchGPRegression(X::AbstractArray,y::AbstractArray;Autotuning::Bool=true,nEpochs::Integer = 10,
                                    kernel=0,noise::Real=1e-3,verbose::Integer=0)
            this = new()
            this.ModelType = Regression
            this.Name = "Non Sparse GP Regression with Gaussian Likelihood"
            initCommon!(this,X,y,noise,1e-16,nEpochs,verbose,Autotuning,1,Adam());
            initFunctions!(this);
            initKernel!(this,kernel);
            this.gnoise = KernelModule.HyperParameter{Float64}(noise,KernelModule.interval(KernelModule.OpenBound{Float64}(zero(Float64)),KernelModule.NullBound{Float64}()))
            
            return this;
    end
end
