"Batch Xtreme Gaussian Process Classifier (no inducing points)"
mutable struct BatchXGPC <: FullBatchModel
    @commonfields
    @functionfields
    @latentfields
    @gaussianparametersfields
    @kernelfields
    "BatchXGPC Constructor"
    function BatchXGPC(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 200,
                                    kernel=0,noise::Float64=1e-3,AutotuningFrequency::Integer=1,
                                    ϵ::Float64=1e-5,μ_init::Array{Float64,1}=[0.0],verbose::Integer=0)
            this = new(X,y)
            this.ModelType = XGPC
            this.Name = "Polya-Gamma Gaussian Process Classifier"
            initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            initGaussian!(this,μ_init);
            initLatentVariables!(this);
            return this;
    end
end
