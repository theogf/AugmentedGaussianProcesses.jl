

mutable struct BatchBSVM <: FullBatchModel
    #Batch Bayesian Support Vector Machine (no sparse inducing points)
    @commonfields
    @functionfields
    @latentfields
    @gaussianparametersfields
    @kernelfields
    
    #Constructor
    function BatchBSVM(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 100,
                                    Kernels=0,γ::Real=1e-3,AutotuningFrequency::Integer=4,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],VerboseLevel::Integer=0)
        this = new()
        this.ModelType = BSVM;
        this.Name = "Full Batch Nonlinear Bayesian SVM"
        initCommon!(this,X,y,γ,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
        initFunctions!(this)
        initKernel!(this,Kernels)
        initGaussian!(this,μ_init)
        initLatentVariables!(this)
        return this
    end
end
