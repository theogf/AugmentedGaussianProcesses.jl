
#Sparse Xtreme Gaussian Process Classifier

mutable struct SparseXGPC <: SparseModel
    @commonfields
    @functionfields
    @latentfields
    @stochasticfields
    @kernelfields
    @sparsefields
    @gaussianparametersfields

    function SparseXGPC(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,BatchSize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    VerboseLevel::Integer=0)
            this = new();
            this.ModelType = XGPC;
            this.Name = "Polya-Gamma Sparse Gaussian Process Classifier";
            initCommon!(this,X,y,noise,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            if Stochastic
                initStochastic!(this,AdaptiveLearningRate,BatchSize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = 1:this.nSamples; this.nSamplesUsed = this.nSamples; this.StochCoeff=1.0;
            end
            initKernel!(this,kernel);
            initSparse!(this,m,OptimizeIndPoints);
            initGaussian!(this,μ_init);
            initLatentVariables!(this);
            return this;
    end
end
