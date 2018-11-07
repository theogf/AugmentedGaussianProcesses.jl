"""Sparse Gaussian Process Classifier with Bayesian SVM likelihood"""
mutable struct SparseBSVM <: SparseModel
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @sparsefields
    @gaussianparametersfields
    α::Vector{Float64}
    θ::Vector{Float64}
    "SparseBSVM Constructor"
    function SparseBSVM(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=10,
                                    verbose::Integer=0)
            this = new()
            this.ModelType = BSVM
            this.Name = "Sparse Nonlinear Bayesian SVM"
            initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            if Stochastic
                initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = 1:this.nSamples; this.nSamplesUsed = this.nSamples;this.StochCoeff=1.0;
            end
            initKernel!(this,kernel);
            initSparse!(this,m,OptimizeIndPoints);
            initGaussian!(this,μ_init);
            this.α = abs.(rand(this.nSamplesUsed))*2;
            this.θ = zero(this.α)
            if this.Stochastic && this.AdaptiveLearningRate
                MCInit!(this)
            end
            return this;
    end
    "Empty constructor for loading models"
    function SparseBSVM()
        this = new()
        this.ModelType = BSVM
        this.Name = "Sparse Nonlinear Bayesian SVM";
        initFunctions!(this)
        return this;
    end
end
