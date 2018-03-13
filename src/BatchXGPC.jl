
#Batch Xtreme Gaussian Process Classifier (no inducing points)

mutable struct BatchXGPC <: FullBatchModel
    @commonfields
    @functionfields
    @latentfields
    @gaussianparametersfields
    @kernelfields
    function BatchXGPC(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 200,
                                    kernel=0,γ::Float64=1e-3,AutotuningFrequency::Integer=10,
                                    ϵ::Float64=1e-5,μ_init::Array{Float64,1}=[0.0],VerboseLevel::Integer=0)
            this = new(X,y)
            this.ModelType = XGPC
            this.Name = "Polya-Gamma Gaussian Process Classifier"
            initCommon!(this,X,y,γ,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            initGaussian!(this,μ_init);
            initLatentVariables!(this);
            return this;
    end
end
