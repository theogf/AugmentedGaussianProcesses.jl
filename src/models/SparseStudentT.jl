"""Sparse Gaussian Process Regression with Student T likelihood"""
mutable struct SparseStudentT{T<:Real} <: SparseModel{T}
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @sparsefields
    @gaussianparametersfields
    ν::T
    α::T
    β::Vector{T}
    θ::Vector{T}
    """SparseStudentT Constructor"""
    function SparseStudentT(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,m::Integer=0,AutotuningFrequency::Integer=1,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    verbose::Integer=0,ν::Real=5.0) where T
            this = new{T}();
            this.ModelType = StudentT;
            this.Name = "Sparse GP Regression with Student-T Likelihood";
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
            this.ν = ν
            this.α = (this.ν+1.0)/2.0
            this.β = abs.(T.(rand(this.nSamplesUsed)))*2;
            this.θ = zero(this.β)
            if this.Stochastic && this.AdaptiveLearningRate
                MCInit!(this)
            end
            return this;
    end
    "Empty constructor for loading models"
    function SparseStudentT{T}() where T
        this = new{T}()
        this.ModelType = StudentT
        this.Name = "Student T Sparse Gaussian Process Regression";
        initFunctions!(this)
        return this;
    end
end
