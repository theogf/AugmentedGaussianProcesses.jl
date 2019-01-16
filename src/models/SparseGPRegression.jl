"""Sparse Gaussian Process Regression with Gaussian Likelihood"""
mutable struct SparseGPRegression{T<:Real} <: SparseModel{T}
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @sparsefields
    @gaussianparametersfields
    gnoise::T
    """Constructor Sparse Gaussian Process Regression with Gaussian Likelihood"""
    function SparseGPRegression(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::T=1.0,τ_s::Integer=100,
                                    kernel=0,noise::T=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    verbose::Integer=0) where {T<:Real}
            this = new{T}();
            this.ModelType = Regression;
            this.Name = "Sparse Gaussian Process Regression with Gaussian Likelihood";
            initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency);
            initFunctions!(this);
            if Stochastic
                initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = 1:this.nSamples; this.nSamplesUsed = this.nSamples; this.StochCoeff=1.0;
            end
            initKernel!(this,kernel);
            initSparse!(this,m,OptimizeIndPoints);
            initGaussian!(this,μ_init);
            if this.Stochastic && this.AdaptiveLearningRate
                MCInit!(this)
            end
            this.gnoise = noise
            return this;
    end
    "Empty constructor for loading models"
    function SparseGPRegression{T}() where T
        this = new{T}()
        this.ModelType = Regression
        this.Name = "Sparse Gaussian Process Regression with Gaussian Likelihood";
        initFunctions!(this)
        return this;
    end
end
