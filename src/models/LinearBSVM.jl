
#Linear Bayesian Support Vector Machines

mutable struct LinearBSVM <: LinearModel
    @commonfields
    @functionfields
    @linearfields
    @latentfields
    @gaussianparametersfields

    @stochasticfields
    #Constructor
    function LinearBSVM(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,Autotuning::Bool=false,optimizer::Optimizer=Adam(),
                                    nEpochs::Integer = 2000,batchsize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    noise::Real=1e-3,AutotuningFrequency::Integer=4, Intercept::Bool=true,ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],
                                    SmoothingWindow::Integer=5, VerboseLevel::Integer=0)
        this = new()
        this.ModelType = BSVM
        this.Name = "Linear Bayesian SVM"
        initCommon!(this,X,y,noise,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
        initFunctions!(this)
        initLinear!(this,Intercept);
        initGaussian!(this,μ_init);
        initLatentVariables!(this);
        if Stochastic
            initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
        else
            this.MBIndices = 1:model.nSamples; this.nSamplesUsed = this.nSamples;this.StochCoeff=1.0;
        end
        return this
    end
end
