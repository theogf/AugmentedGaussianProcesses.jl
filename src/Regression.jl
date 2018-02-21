
#Batch Gaussian Process Regression (no inducing points)

mutable struct GPRegression <: FullBatchModel
    @commonfields
    @functionfields
    @kernelfields
    function BatchXGPC(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 10,
                                    Kernels=0,γ::Float64=1e-3,VerboseLevel::Integer=0)
            this = new(X,y)
            this.ModelType = Regression
            this.Name = "Gaussian Process Regression"
            initCommon!(this,X,y,γ,0,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,Kernels);
            return this;
    end
end
