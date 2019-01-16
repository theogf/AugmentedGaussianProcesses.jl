
#Online Gaussian Process Classifier

mutable struct OnlineXGPC{T<:Real} <: OnlineGPModel{T}
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @gaussianparametersfields
    @onlinefields
    c::Vector{T}
    θ::Vector{T}
    function OnlineXGPC(X::AbstractArray{T},y::AbstractArray;kmeansalg::KMeansAlg=StreamOnline(),Sequential::Bool=false,AdaptiveLearningRate::Bool=false,
                                    Autotuning::Bool=false,OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    verbose::Integer=0) where {T<:Real}
            this = new{T}();
            this.ModelType = XGPC;
            this.Name = "Online Sparse Gaussian Process Classification";
            initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency);
            initOnline!(this,kmeansalg,Sequential,m)
            initFunctions!(this);
            initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            initGaussian!(this,μ_init);
            initKernel!(this,kernel); this.nFeatures = this.m
            this.c = abs.(rand(this.nSamplesUsed))*2;
            this.θ = zero(this.c)
            return this;
    end
end
