"Batch Bayesian Support Vector Machine (no sparse inducing points)"
mutable struct BatchBSVM <: FullBatchModel
    @commonfields
    @functionfields
    @latentfields
    @gaussianparametersfields
    @kernelfields

    "BatchBSVM Constructor"
    function BatchBSVM(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 100,
                                    kernel=0,noise::Real=1e-3,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],verbose::Integer=0)
        this = new()
        this.ModelType = BSVM;
        this.Name = "Full Batch Nonlinear Bayesian SVM"
        initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
        initFunctions!(this)
        initKernel!(this,kernel)
        initGaussian!(this,μ_init)
        initLatentVariables!(this)
        return this
    end
end
