
#Batch Xtreme Gaussian Process Classifier (no inducing points)

mutable struct SparseMultiClass <: SparseModel
    @commonfields
    @functionfields
    @multiclassfields
    @stochasticfields
    @kernelfields
    @multiclass_sparsefields
    function SparseMultiClass(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,BatchSize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    VerboseLevel::Integer=0)
            Y,y_map,y_class = one_of_K_mapping(y)
            this = new()
            this.ModelType = MultiClassModel
            this.Name = "MultiClass Gaussian Process Classifier"
            initCommon!(this,X,y,noise,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            if Stochastic
                initStochastic!(this,AdaptiveLearningRate,BatchSize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = 1:this.nSamples; this.nSamplesUsed = this.nSamples; this.StochCoeff=1.0;
            end
            initKernel!(this,kernel);
            initMultiClass!(this,Y,y_class,y_map,μ_init);
            initMultiClassSparse!(this,m,optimizeIndPoints)
            return this;
    end
end
