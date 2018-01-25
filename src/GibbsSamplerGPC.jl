

mutable struct GibbsSamplerGPC <: FullBatchModel
    #Batch Xtreme Gaussian Process Classifier (no inducing points)
    @commonfields
    @functionfields
    @latentfields
    @gaussianparametersfields
    
    @kernelfields
    @samplingfields
    function GibbsSamplerGPC(X::AbstractArray,y::AbstractArray;burninsamples::Integer = 200, samplefrequency::Integer=100,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 200,
                                    Kernels=0,γ::Float64=1e-3,AutotuningFrequency::Integer=10,
                                    ϵ::Float64=1e-5,μ_init::Array{Float64,1}=[0.0],VerboseLevel::Integer=0)
            this = new(X,y)
            this.ModelType = XGPC
            this.Name = "Polya-Gamma Gaussian Process Classifier by sampling"
            initCommon!(this,X,y,γ,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,Kernels);
            initGaussian!(this,μ_init);
            initLatentVariables!(this);
            initSampling!(this,burninsamples,samplefrequency)
            return this;
    end

end
