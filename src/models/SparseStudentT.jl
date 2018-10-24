"""Sparse Student T Gaussian Process Regression
"""
mutable struct SparseStudentT <: SparseModel
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @sparsefields
    @gaussianparametersfields
    ν::Float64
    α::Float64
    β::Vector{Float64}
    θ::Vector{Float64}
    "SparseStudentT Constructor"
    function SparseStudentT(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    verbose::Integer=0,ν::Float64=5.0)
            this = new();
            this.ModelType = StudentT;
            this.Name = "Student T Sparse Gaussian Process Regression";
            initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
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
            this.β = abs.(rand(this.nSamplesUsed))*2;
            this.θ = zero(this.β)
            if this.Stochastic && this.AdaptiveLearningRate
                MCInit!(this)
            end
            return this;
    end
    "Empty constructor for loading models"
    function SparseStudentT()
        this = new()
        this.ModelType = StudentT
        this.Name = "Student T Sparse Gaussian Process Regression";
        initFunctions!(this)
        return this;
    end
end
