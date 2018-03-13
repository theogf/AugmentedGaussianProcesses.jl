
#Sparse Bayesian Support Vector Machines (with inducing points)

mutable struct SparseBSVM <: SparseModel
    @commonfields
    @functionfields
    @latentfields
    @stochasticfields
    @kernelfields
    @sparsefields

    @gaussianparametersfields
    #Constructor
    function SparseBSVM(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,BatchSize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,γ::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=5,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=10,
                                    VerboseLevel::Integer=0)
            this = new()
            this.ModelType = BSVM
            this.Name = "Sparse Nonlinear Bayesian SVM"
            initCommon!(this,X,y,γ,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            initSparse!(this,m,OptimizeIndPoints);
            initGaussian!(this,μ_init);
            initLatentVariables!(this);
            if Stochastic
                initStochastic!(this,AdaptiveLearningRate,BatchSize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = 1:this.nSamples; this.nSamplesUsed = this.nSamples;this.StochCoeff=1.0;
            end
            return this;
    end
end
